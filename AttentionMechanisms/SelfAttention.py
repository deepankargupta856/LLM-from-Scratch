import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class SelfAttention(nn.Module):
    """
    Simple self-attention module.

    Args:
        input_dim:  Dimensionality of the input features.
        output_dim: Dimensionality of the query/key/value space.
        bias:       Whether to include bias terms in the linear projections.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.query_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.key_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.value_proj = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for self-attention.

        Args:
            x: Tensor of shape (sequence_length, input_dim)

        Returns:
            context: Tensor of shape (sequence_length, output_dim)
        """
        keys = self.key_proj(x)            # (seq_len, output_dim)
        queries = self.query_proj(x)       # (seq_len, output_dim)
        values = self.value_proj(x)        # (seq_len, output_dim)

        # Compute raw attention scores
        # (seq_len, output_dim) @ (output_dim, seq_len) -> (seq_len, seq_len)
        scores = queries @ keys.transpose(0, 1)

        # Scale and normalize
        dk = keys.size(-1)
        scaled_scores = scores / (dk**0.5)
        attn_weights = torch.softmax(scaled_scores, dim=-1)

        # Weighted sum of values
        context = attn_weights @ values    # (seq_len, output_dim)
        return context


def manual_self_attention_demo() -> None:
    """
    Manual step-by-step computation using torch.nn.Parameter
    to verify that SelfAttention matches hand-rolled results.
    """
    # Example inputs (6 tokens × 3 features)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])

    seq_len, feature_dim = inputs.shape
    proj_dim = 2

    # Fix the random seed for reproducibility
    torch.manual_seed(123)

    # Create random but fixed projection matrices
    W_query = nn.Parameter(torch.rand(feature_dim, proj_dim), requires_grad=False)
    W_key = nn.Parameter(torch.rand(feature_dim, proj_dim), requires_grad=False)
    W_value = nn.Parameter(torch.rand(feature_dim, proj_dim), requires_grad=False)

    # Pick the second token as a test query
    query_vector = inputs[1] @ W_query

    # Compute all projections
    all_queries = inputs @ W_query    # (6, 2)
    all_keys = inputs @ W_key         # (6, 2)
    all_values = inputs @ W_value     # (6, 2)

    # Attention scores for the second token
    scores = query_vector @ all_keys.transpose(0, 1)
    dk = all_keys.size(-1)
    weights = torch.softmax(scores / (dk**0.5), dim=-1)

    print("=== Manual Self-Attention Debug ===")
    print("Query (token 2):", query_vector)
    print("All Keys shape:", all_keys.shape)
    print("All Values shape:", all_values.shape)
    print("Attention scores:", scores)
    print("Attention weights:", weights)

    # Compare against module output
    torch.manual_seed(789)
    attention_module = SelfAttention(input_dim=feature_dim, output_dim=proj_dim)
    module_output = attention_module(inputs)
    print("Module output:", module_output)


def main() -> None:
    """
    Entry point for running the manual self-attention demo.
    """
    manual_self_attention_demo()


if __name__ == "__main__":
    main()
