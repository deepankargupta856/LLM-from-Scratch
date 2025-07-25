import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class CausalSelfAttention(nn.Module):
    """
    Causal (autoregressive) self-attention module with dropout.

    Masks out future tokens so each position only attends to itself and previous positions.

    Args:
        input_dim:        Number of features per token.
        proj_dim:         Dimensionality of the queries/keys/values.
        max_context_len:  Maximum sequence length (for creating the causal mask).
        dropout_prob:     Dropout probability applied to attention weights.
        bias:             Whether to include bias terms in the linear projections.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        max_context_len: int,
        dropout_prob: float = 0.0,
        bias: bool = False
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(input_dim, proj_dim, bias=bias)
        self.key_proj = nn.Linear(input_dim, proj_dim, bias=bias)
        self.value_proj = nn.Linear(input_dim, proj_dim, bias=bias)

        # Dropout on attention weights
        self.attn_dropout = nn.Dropout(dropout_prob)

        # Causal mask: upper triangle (1 above diag) set to True
        mask = torch.triu(torch.ones(max_context_len, max_context_len), diagonal=1)
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, proj_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Project to queries, keys, values
        queries = self.query_proj(x)   # (B, T, D)
        keys = self.key_proj(x)        # (B, T, D)
        values = self.value_proj(x)    # (B, T, D)

        # Raw attention scores: (B, T, D) @ (B, D, T) -> (B, T, T)
        # Use transpose on the last two dims of keys
        scores = queries @ keys.transpose(-2, -1)

        # Apply causal mask: set future positions to -inf
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        # Scale and softmax
        scale = proj_dim = self.proj_dim
        scores = scores / (scale ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Dropout on attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (B, T, T) @ (B, T, D) -> (B, T, D)
        context = attn_weights @ values
        return context


def demo_causal_attention() -> None:
    """
    Demonstrates the CausalSelfAttention on a dummy batch of size 2,
    sequence length = 6, feature dimension = 3, projection dim = 2.
    """
    # Dummy input: 6 tokens × 3 features
    tokens = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])

    # Create a batch of size 2
    batch = torch.stack([tokens, tokens], dim=0)  # (2, 6, 3)

    batch_size, seq_len, feature_dim = batch.shape
    proj_dim = 2
    dropout = 0.0

    print(f"Input batch shape: {batch.shape}")

    # Seed for reproducibility
    torch.manual_seed(123)

    # Instantiate and run
    attention = CausalSelfAttention(
        input_dim=feature_dim,
        proj_dim=proj_dim,
        max_context_len=seq_len,
        dropout_prob=dropout,
        bias=False
    )
    output = attention(batch)

    print(f"Output context vectors shape: {output.shape}")
    print(output)


def main() -> None:
    demo_causal_attention()


if __name__ == "__main__":
    main()
