#!/usr/bin/env python3
"""
This module defines a Transformer block consisting of masked multi-head
self-attention followed by a position-wise feed-forward network, each
with residual connections and layer normalization.
"""

import torch
import torch.nn as nn

from AttentionMechanisms.MaskedMultiHeadAttention import MultiHeadAttention
from GPTarchitechture.FeedForward import FeedForward
from GPTarchitechture.LayerNormalization import LayerNorm


class TransformerBlock(nn.Module):
    """
    Single Transformer block.

    Combines:
        1) LayerNorm → Masked Multi-Head Self-Attention → Dropout → Residual
        2) LayerNorm → Feed-Forward Network → Dropout → Residual
    """
    def __init__(self, cfg: dict):
        """
        Initialize the Transformer block.

        Args:
            cfg (dict): Configuration dictionary containing:
                - "emb_dim" (int): Embedding dimension.
                - "context_length" (int): Maximum sequence length.
                - "n_heads" (int): Number of attention heads.
                - "drop_rate" (float): Dropout probability.
                - "qkv_bias" (bool): Whether to use bias in Q/K/V projections.
        """
        super().__init__()

        # Multi-head self-attention sub-layer
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Position-wise feed-forward network sub-layer
        self.feed_forward = FeedForward(cfg)

        # Layer normalization modules for pre-norm
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout for residual branches
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # --- Attention sub-layer with residual connection ---
        # Save input for residual shortcut
        residual = x

        # Pre-norm
        x = self.norm1(x)

        # Masked multi-head self-attention
        x = self.attention(x)

        # Dropout and add residual
        x = self.dropout(x)
        x = x + residual

        # --- Feed-forward sub-layer with residual connection ---
        residual = x

        # Pre-norm
        x = self.norm2(x)

        # Position-wise feed-forward network
        x = self.feed_forward(x)

        # Dropout and add residual
        x = self.dropout(x)
        x = x + residual

        return x


def main():
    """Demonstrate the TransformerBlock with a random tensor."""
    # Example GPT-like configuration
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # For reproducibility
    torch.manual_seed(123)

    # Create a random input: batch_size=2, seq_len=4, emb_dim=768
    x = torch.rand(2, 4, GPT_CONFIG_124M["emb_dim"])

    # Instantiate and apply the Transformer block
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    # Display shapes
    print("Input shape: ", x.shape)     # Expected: torch.Size([2, 4, 768])
    print("Output shape:", output.shape)  # Expected: torch.Size([2, 4, 768])


if __name__ == "__main__":
    main()
