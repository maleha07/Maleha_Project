from model import TinyUNet

model = TinyUNet()

total = sum(p.numel() for p in model.parameters())
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(model)
print(f"\nTotal parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
