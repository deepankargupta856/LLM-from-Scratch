#!/usr/bin/env python3
"""
This module implements a simple feed‑forward network block as used in GPT-style
transformers, with an expansion, nonlinearity, and contraction.
"""

import torch
import torch.nn as nn

from GPTarchitechture.Activations import GELU
class FeedForward(nn.Module):
    """
    Transformer-style feed-forward network.

    Consists of:
        1) Linear expansion: emb_dim → 4 * emb_dim
        2) GELU activation
        3) Linear projection: 4 * emb_dim → emb_dim
    """
    def __init__(self, cfg: dict):
        """
        Initialize FeedForward block.

        Args:
            cfg (dict): Configuration dictionary containing:
                - "emb_dim" (int): Embedding dimension.
        """
        super().__init__()
        emb_dim = cfg["emb_dim"]

        self.layers = nn.Sequential(
            # Expansion layer
            nn.Linear(emb_dim, 4 * emb_dim),
            # Nonlinear activation
            GELU(),
            # Projection back to embedding dimension
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.layers(x)


def main():
    """Demonstrate the FeedForward block with a random tensor."""
    # GPT‑style configuration
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size (unused here)
        "context_length": 1024, # Context length (unused here)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads (unused here)
        "n_layers": 12,         # Number of layers (unused here)
        "drop_rate": 0.1,       # Dropout rate (unused here)
        "qkv_bias": False       # QKV bias flag (unused here)
    }

    # Instantiate the feed-forward network
    ffn = FeedForward(GPT_CONFIG_124M)

    # Create a random input: batch_size=2, seq_len=3, emb_dim=768
    x = torch.rand(2, 3, GPT_CONFIG_124M["emb_dim"])

    # Forward pass
    out = ffn(x)

    # Print the output shape
    print("Output shape:", out.shape)  # Expected: torch.Size([2, 3, 768])


if __name__ == "__main__":
    main()
