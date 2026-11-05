import torch
import pytest
from src.model.mlp import ASLLandmarkMLP

def test_mlp_architecture():
    model = ASLLandmarkMLP()

    # Test trainable parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 18524, f"Expected 18524 parameters, got {total_params}"

    # Test forward pass shape and dtype
    batch_size = 4
    x = torch.randn(batch_size, 63)
    out = model(x)

    assert out.shape == (batch_size, 28)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
