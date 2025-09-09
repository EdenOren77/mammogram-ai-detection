from src.model import SimpleCNN
import torch

model = SimpleCNN()

dummy_input = torch.randn(1, 1, 487, 552)
output = model(dummy_input)
print("Output shape:", output.shape)