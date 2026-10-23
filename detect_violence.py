"""WASP local violence-video scanner (prototype).

Loads the visionlab-ai/school-violence-detection-models X3D-M checkpoint and
scans ALL of a video using overlapping temporal windows. This is still a
research prototype; evaluate on representative workplace footage before use.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.hub import x3d_m

NUM_FRAMES = 16
FRAME_SIZE = 224
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
DEFAULT_THRESHOLD = 0.40


class ViolenceX3D(nn.Module):
    """Wrapper matching the downloaded checkpoint's backbone.* state keys."""

    def __init__(self, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.backbone = x3d_m(pretrained=False)
        self.backbone.blocks[5].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.backbone(video)


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Only use weights_only=False for the specific checkpoint downloaded from
    # the intended repository, because its metadata includes pickled objects.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError("Expected checkpoint dictionary with a 'model' state_dict")
    model = ViolenceX3D(dropout=0.3, num_classes=2)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def make_window_starts(duration: float, window_seconds: float, stride_seconds: float) -> list[float]:
    if duration <= window_seconds:
        return [0.0]
    last_start = max(0.0, duration - window_seconds)
    starts = list(np.arange(0.0, last_start + 1e-6, stride_seconds))
    if not starts or last_start - starts[-1] > 1e-3:
        starts.append(last_start)
    # Round to milliseconds and deduplicate overlapping last start.
    return list(dict.fromkeys(round(float(s), 3) for s in starts))


def sample_window(cap: cv2.VideoCapture, start_s: float, window_s: float,
                  fps: float, total_frames: int) -> torch.Tensor:
    start_idx = max(0, min(total_frames - 1, int(round(start_s * fps))))
    end_idx = max(start_idx, min(total_frames - 1,
                                int(round((start_s + window_s) * fps)) - 1))
    if end_idx == start_idx:
        indices = np.full(NUM_FRAMES, start_idx, dtype=int)
    else:
        indices = np.linspace(start_idx, end_idx, NUM_FRAMES).round().astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # This matches the simple 224x224 resize assumption in the model card.
        # If the original training transform differs, preprocessing must be aligned.
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def scan_video(model: nn.Module, video_path: Path, device: torch.device,
               threshold: float, window_seconds: float, stride_seconds: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    if total_frames < 1:
        cap.release()
        raise RuntimeError("Video has no readable frames")

    duration = total_frames / fps
    actual_window = min(window_seconds, duration)
    starts = make_window_starts(duration, actual_window, stride_seconds)
    results = []
    total_inference_ms = 0.0
    total_preprocess_ms = 0.0

    try:
        for number, start_s in enumerate(starts, start=1):
            prep_start = time.perf_counter()
            video = sample_window(cap, start_s, actual_window, fps, total_frames)
            total_preprocess_ms += (time.perf_counter() - prep_start) * 1000
            video = video.to(device)

            with torch.inference_mode():
                # Warm up only the first window, not counted as model timing.
                if number == 1:
                    with autocast_context(device):
                        _ = model(video)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                t0 = time.perf_counter()
                with autocast_context(device):
                    logits = model(video)
                    probs = F.softmax(logits.float(), dim=1)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t0) * 1000

            total_inference_ms += infer_ms
            # Model card labels: class 0 = non-violent, class 1 = violent.
            p_violence = float(probs[1].item())
            results.append((start_s, min(duration, start_s + actual_window), p_violence, infer_ms))
            print(f"Window {number:03d}/{len(starts):03d} "
                  f"[{start_s:6.2f}s–{min(duration, start_s + actual_window):6.2f}s] "
                  f"violence={p_violence:.4f} ({p_violence*100:5.1f}%) "
                  f"inference={infer_ms:.1f} ms")
    finally:
        cap.release()

    best = max(results, key=lambda r: r[2])
    print("\n=== WASP full-video scan ===")
    print("Video:", video_path.resolve())
    print(f"Duration: {duration:.2f}s | FPS: {fps:.2f} | Frames: {total_frames}")
    print(f"Windows scanned: {len(results)} | window: {actual_window:.2f}s | stride: {stride_seconds:.2f}s")
    print(f"Highest violence score: {best[2]:.4f} ({best[2]*100:.1f}%) "
          f"at {best[0]:.2f}s–{best[1]:.2f}s")
    print(f"Threshold: {threshold:.2f}")
    print("Decision:", "POTENTIAL VIOLENCE — admin review recommended" if best[2] >= threshold else "NO WINDOW REACHED THRESHOLD")
    print(f"Total model inference: {total_inference_ms:.1f} ms across all windows")
    print(f"Total frame decode/preprocessing: {total_preprocess_ms:.1f} ms")
    print("NOTE: low scores on known violent footage mean the checkpoint/preprocessing/domain must be investigated; do not simply trust the label.")


def main() -> None:
    parser = argparse.ArgumentParser(description="WASP overlapping-window violence scan")
    parser.add_argument("video", help="Path to a video file")
    parser.add_argument("--checkpoint", default="models/final/final_x3d_realtime.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window-seconds", type=float, default=4.0,
                        help="Clip duration for each model prediction (default 4s)")
    parser.add_argument("--stride-seconds", type=float, default=1.0,
                        help="Advance between windows; smaller catches shorter events but costs more (default 1s)")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        parser.error("window and stride must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    model = load_model(Path(args.checkpoint), device)
    print("Loaded checkpoint:", Path(args.checkpoint).resolve())
    scan_video(model, Path(args.video), device, args.threshold,
               args.window_seconds, args.stride_seconds)


if __name__ == "__main__":
    main()
