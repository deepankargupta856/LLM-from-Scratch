#!/usr/bin/env python3
"""
This module provides a simple LayerNorm implementation and demonstrates
its usage alongside a sample nn.Sequential block.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.

    Attributes:
        eps (float): Small constant to avoid division by zero.
        scale (nn.Parameter): Learnable scale (gamma) parameter.
        shift (nn.Parameter): Learnable shift (beta) parameter.
    """
    def __init__(self, emb_dim: int):
        """
        Initialize the LayerNorm module.

        Args:
            emb_dim (int): Dimensionality of the embeddings to normalize.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization over the last dimension of the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim).

        Returns:
            torch.Tensor: The normalized, scaled, and shifted output.
        """
        # Compute mean and variance along the embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.scale * norm_x + self.shift


def main():
    """Entry point for the module; runs a small demonstration."""
    # Reproducibility
    torch.manual_seed(123)

    # Create a random batch of 2 samples, each of dimension 5
    batch_example = torch.randn(2, 5)

    # Define a simple feed-forward layer for demonstration
    mlp = nn.Sequential(
        nn.Linear(5, 6),
        nn.ReLU()
    )
    out_mlp = mlp(batch_example)
    print("Output of nn.Sequential block:\n", out_mlp)

    # Apply custom layer normalization
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)

    # Print mean and variance after normalization to verify behavior
    torch.set_printoptions(sci_mode=False)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("\nLayerNorm output mean:\n", mean)
    print("LayerNorm output variance:\n", var)


if __name__ == "__main__":
    main()
