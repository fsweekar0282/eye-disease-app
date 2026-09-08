import torch

sd = torch.load("models/eye_disease_model_v4.pth", map_location="cpu")

print("=== HEAD KEYS ===")
for k, v in sd.items():
    if k.startswith("head"):
        print(f"{k:45} {tuple(v.shape)}")

print("\n=== BACKBONE TAIL (non-block keys) ===")
for k, v in sd.items():
    if k.startswith("backbone") and ".blocks." not in k:
        print(f"{k:45} {tuple(v.shape)}")