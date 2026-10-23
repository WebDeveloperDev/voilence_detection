from pathlib import Path
import torch

files = list(Path("models").rglob("final_x3d_realtime.pt"))

if not files:
    print("ERROR: Checkpoint not found under the models folder.")
    print("Check the file location shown by Get-ChildItem.")
    raise SystemExit(1)

path = files[0]
print("Checkpoint:", path.resolve())
print("File size:", round(path.stat().st_size / (1024 * 1024), 2), "MB")

# Use only with the checkpoint downloaded from the model repository.
checkpoint = torch.load(
    str(path),
    map_location="cpu",
    weights_only=False
)

print("Checkpoint type:", type(checkpoint).__name__)

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys())[:30])

    state_dict = checkpoint
    for key in ("state_dict", "model", "model_state_dict", "weights"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            print("Using state dictionary:", key)
            break

    tensors = [
        (key, value)
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    ]

    print("Tensor count:", len(tensors))
    print("\nFirst 35 parameter names and shapes:")

    for name, tensor in tensors[:35]:
        print(f"{name}: {tuple(tensor.shape)}")
else:
    print("Unexpected checkpoint format; do not load it into the model yet.")
