"""Quick test: can PyTorch train on this Mac's GPU (MPS)? Prints each step as it finishes."""

import platform
import time

import torch
import torchvision

print(f"macOS {platform.mac_ver()[0]}, Python {platform.python_version()}, "
      f"torch {torch.__version__}, torchvision {torchvision.__version__}", flush=True)
print(f"MPS built: {torch.backends.mps.is_built()}, available: {torch.backends.mps.is_available()}", flush=True)

for device in ["cpu", "mps"]:
    if device == "mps" and not torch.backends.mps.is_available():
        continue
    print(f"\n--- {device} ---", flush=True)
    t = time.time()
    a = torch.randn(1024, 1024, device=device)
    (a @ a).sum().item()
    print(f"1. matrix multiply OK ({time.time() - t:.1f} s)", flush=True)

    m = torchvision.models.resnet50().to(device)
    x = torch.randn(8, 3, 224, 224, device=device)
    t = time.time()
    with torch.no_grad():
        m(x).sum().item()
    print(f"2. ResNet-50 forward pass OK ({time.time() - t:.1f} s)", flush=True)

    t = time.time()
    m(x).sum().backward()
    if device == "mps":
        torch.mps.synchronize()
    print(f"3. ResNet-50 backward pass OK ({time.time() - t:.1f} s)", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    t = time.time()
    opt.step()
    if device == "mps":
        torch.mps.synchronize()
    print(f"4. optimizer step OK ({time.time() - t:.1f} s)", flush=True)

print("\nAll steps finished.")
