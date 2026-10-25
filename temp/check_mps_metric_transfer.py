"""Compare CPU and MPS metric transfers using tiny, known nonzero values."""

import sys

import torch


print(f"Python: {sys.executable}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)
if not torch.backends.mps.is_available():
    raise SystemExit("This diagnostic requires MPS access.")

for device in ("cpu", "mps"):
    delta = torch.tensor(
        [[[-0.003, 0.004, -0.012], [0, -0.003, 0.004]],
         [[0.001, 0, 0], [0, 0, -0.002]],
         [[0, -0.030, 0], [0.040, 0, 0]]],
        dtype=torch.float32, device=device,
    )
    distances = torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)
    combined = distances.to(device="cpu", dtype=torch.float64)
    separate = distances.cpu().double()
    print(f"{device}: distances={distances.cpu().tolist()}", flush=True)
    print(f"{device}: combined transfer/cast={combined.tolist()}, mean={combined.mean().item():.12g}", flush=True)
    print(f"{device}: separate transfer/cast={separate.tolist()}, mean={separate.mean().item():.12g}", flush=True)
    absolute_errors = delta.abs()
    print(f"{device}: MAE combined={absolute_errors.to(device='cpu', dtype=torch.float64).mean().item():.12g}", flush=True)
    print(f"{device}: MAE separate={absolute_errors.cpu().double().mean().item():.12g}", flush=True)

print(f"Expected mean distance: {0.091 / 6:.12g} m", flush=True)
