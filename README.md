# WASP — Local Violence Detection Prototype

WASP is a workplace-safety application. This repository contains an **offline video-scanning prototype** that classifies sampled clips from a video as potentially violent or non-violent. The intended deployment is local processing on an organization's admin computer; the mobile app's manual SOS feature is separate from this detector.

> **Prototype / safety notice:** This is an assistive screening tool, not a guaranteed violence detector. It can miss incidents or flag normal activity. Validate it on representative, properly authorized workplace footage before using it operationally. Keep a human admin in the alert-review loop.

## Features

- Runs locally; video is not uploaded by this script.
- Uses an X3D-M-based checkpoint fine-tuned for two-class violence classification.
- Scans the **entire video** using overlapping temporal windows rather than examining only the end.
- Prints a violence score for each window, the highest score, the configured threshold, inference time, and frame-preprocessing time.
- Uses an NVIDIA GPU with CUDA when available; otherwise, it falls back to CPU (likely much slower).

## Current implementation

| Setting | Value |
| --- | --- |
| Model | X3D-M with a two-class classifier head |
| Checkpoint | `models/final/final_x3d_realtime.pt` |
| Frames sampled per window | 16 |
| Default window length | 4 seconds |
| Default window stride | 1 second (overlapping windows) |
| Default violence threshold | `0.40` |
| Input frame size | 224 × 224 |
| Input normalization | RGB; channel mean `0.45`, standard deviation `0.225` |
| Preferred device | CUDA-enabled NVIDIA GPU |

The script treats class index `0` as non-violent and class index `1` as violent, following the checkpoint's model-card labels. For each window it reports the class-1 probability. The final decision is based on the **highest window score** in the video.

## Requirements

The setup below targets the tested Windows environment:

- Windows 10/11 and PowerShell
- Python 3.11 (recommended for this project)
- NVIDIA GPU and a compatible, installed NVIDIA driver for CUDA inference
- Internet access for installing dependencies and downloading the checkpoint
- Enough disk space for Python packages, the checkpoint, and test videos

Example hardware used during development: NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM, Ryzen 5 CPU, and 24 GB RAM. Other hardware may have different speed and memory characteristics.

## Setup on Windows

Run these commands from PowerShell. Replace the project path if yours is different.

### 1. Open the project and activate the virtual environment

```powershell
Set-Location 'D:\dev\projects\voilence_detection'

# Only create the environment if .venv does not already exist:
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
```

The Python version should show `3.11.x`. If PowerShell blocks activation, run the following **in the current PowerShell session only**, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA support

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install video-processing dependencies

```powershell
python -m pip install pytorchvideo==0.1.5 opencv-python huggingface_hub pyyaml
```

### 4. Verify the installation

```powershell
python -c "import torch, cv2, pytorchvideo; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected; using CPU'); print('OpenCV:', cv2.__version__); print('PyTorchVideo: OK')"
```

For GPU inference, `CUDA available` should be `True` and the NVIDIA GPU name should be printed. If it is `False`, the script can fall back to CPU, but performance will usually be considerably slower. Resolve the PyTorch/driver setup before evaluating latency.

### 5. Download the pretrained checkpoint

Run from the project root, with `.venv` activated:

```powershell
python -c "from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='visionlab-ai/school-violence-detection-models', filename='final/final_x3d_realtime.pt', local_dir='models'); print('Model downloaded to:', p)"
```

Confirm that this file exists:

```powershell
Test-Path .\models\final\final_x3d_realtime.pt
```

It should return `True`.

### 6. Confirm the project files

The relevant files should look like this:

```text
voilence_detection/
├── .venv/
├── models/
│   └── final/
│       └── final_x3d_realtime.pt
├── detect_violence.py
└── README.md
```

`inspect_checkpoint.py`, if you created it during setup, is optional and only used to inspect the checkpoint.

## Run the detector

### Scan a video

Use a local video file that OpenCV can read. Keep quotation marks around paths containing spaces:

```powershell
python .\detect_violence.py "D:\Videos\workplace_test.mp4"
```

The script loads the model, scans the video, and reports a score for each window. Example output format (values below are illustrative):

```text
Device: cuda
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Loaded checkpoint: ...\models\final\final_x3d_realtime.pt
Window 001/006 [  0.00s–  4.00s] violence=0.7310 ( 73.1%) inference=...
...
=== WASP full-video scan ===
Video: ...
Duration: ... | FPS: ... | Frames: ...
Windows scanned: ... | window: 4.00s | stride: 1.00s
Highest violence score: ... at ...s–...s
Threshold: 0.40
Decision: POTENTIAL VIOLENCE — admin review recommended
Total model inference: ...
Total frame decode/preprocessing: ...
```

Actual output and timing depend on the input video and hardware. A high score is a model prediction, not proof that violence occurred.

### Change the threshold

```powershell
python .\detect_violence.py "D:\Videos\workplace_test.mp4" --threshold 0.50
```

- A **lower** threshold makes it easier for a window to trigger the potential-violence decision. This may catch more incidents but can increase false alarms.
- A **higher** threshold reduces alerts but may miss more incidents.
- The default threshold is `0.40`, as configured for this prototype. Do not choose a deployment threshold based on a single video; calibrate it on a separate, representative validation set.

### Change window length or stride

```powershell
python .\detect_violence.py "D:\Videos\workplace_test.mp4" --window-seconds 4 --stride-seconds 1
```

- `--window-seconds`: duration covered by each model window; default `4.0` seconds.
- `--stride-seconds`: time between the starts of windows; default `1.0` second.
- A smaller stride scans more overlapping windows and may improve temporal coverage, but increases runtime.
- A larger stride is faster but may skip short events between windows.

### Use a different checkpoint path

```powershell
python .\detect_violence.py "D:\Videos\workplace_test.mp4" --checkpoint "D:\models\final_x3d_realtime.pt"
```

## How it works

1. Open the video with OpenCV and read its frame count and frame rate.
2. Divide the full video into overlapping windows. The default is four seconds per window, with a one-second stride; the final window is included so the end of the video is covered.
3. Sample 16 frames evenly across each window.
4. Convert frames from BGR to RGB, resize them to 224 × 224, normalize them, and arrange them into the model's expected tensor format.
5. Run inference on CUDA when available, using half-precision autocasting on the GPU.
6. Print each window's violence probability and compare the maximum probability across all windows with the configured threshold.

This is an **offline recorded-video scanner**. It does not currently connect to RTSP/IP cameras, run continuous multi-camera monitoring, save incident clips, send notifications, or integrate with the WASP admin dashboard. Those require separate components and testing.

## Testing and validation guidance

Before using the detector to support workplace-safety decisions:

1. Test on a range of representative workplace clips, including both violent and non-violent activity.
2. Include confusing negatives such as hugs, play, dancing, sports, crowded movement, arguments without physical violence, and accidental contact.
3. Use recordings from different sessions/cameras for training, validation, and testing where possible; do not split nearly identical frames across these sets.
4. Measure recall (missed incidents), precision (how many alerts are genuine), false alerts per camera-hour, and time from incident onset to detection—not accuracy alone.
5. Have an authorized human admin review alerts. Keep manual SOS available independently of video-model results.

The checkpoint was trained on a particular violence-video dataset. Its published scores do not guarantee accuracy on workplace CCTV, and this prototype should not be treated as a certified or safety-critical detector.

## Troubleshooting

### `CUDA available: False`

Check that the NVIDIA driver is installed and `nvidia-smi` works. Reinstall the matching CUDA-enabled PyTorch packages inside the activated `.venv`, then rerun the verification command. Make sure you are using the virtual environment's Python (`python`, not another system Python).

### `Checkpoint not found`

Run the command from the project root, or provide the full path through `--checkpoint`. Confirm that `models\final\final_x3d_realtime.pt` exists.

### OpenCV cannot open the video

Check that the path is correct, quote paths with spaces, and try a common container/codec such as MP4 with H.264 video. Some camera-specific codecs may need conversion before OpenCV can read them.

### Scores are low on a known violent clip

Do not assume that the clip is safe. Confirm that the event appears in the video and is sampled by the windows, check the checkpoint and preprocessing, and test multiple clips. If the problem persists, evaluate another checkpoint or fine-tune on representative workplace footage. Do not simply lower the threshold without measuring false alarms.

### Too many alerts

Evaluate false positives on representative non-violent recordings and tune the threshold on a validation set. Temporal aggregation can be added later, but should be tested carefully so it does not delay or suppress real incidents.

## Privacy and checkpoint safety

- This script performs inference locally and does not upload video, but your wider application may have separate network behavior.
- Only analyze footage you are authorized to access and process. Follow applicable workplace privacy, consent, security, and retention policies.
- The checkpoint is loaded with Python/PyTorch deserialization settings intended for this specific downloaded artifact. Do not substitute arbitrary or untrusted `.pt` files.

## References

- Pretrained model repository: <https://huggingface.co/visionlab-ai/school-violence-detection-models>
- Reference article: <https://medium.com/@nidhi.dubagunta/using-machine-learning-for-real-time-detection-of-violence-in-video-footage-82feeede3150>
- PyTorch installation instructions: <https://pytorch.org/get-started/locally/>
