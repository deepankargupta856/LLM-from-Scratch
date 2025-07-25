import torch
import torch.nn as nn
from AttentionMechanisms.CausalAttention import CausalSelfAttention


class MultiHeadAttentionWrapper(nn.Module):
    """
    Naïve multi‑head attention that instantiates multiple single‑head
    causal self‑attention modules and concatenates their outputs.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each head and concatenate on the feature dimension
        head_outputs = [head(x) for head in self.heads]
        return torch.cat(head_outputs, dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Efficient multi‑head attention via weight splitting.
    Projects input into combined Q/K/V of size d_out, then
    splits into num_heads heads internally.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Combined projections for queries, keys, values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final output projection
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Tensor of shape (batch_size, seq_len, d_out)
        """
        b, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_query(x)  # (b, seq_len, d_out)
        K = self.W_key(x)
        V = self.W_value(x)

        # Reshape and split heads: (b, num_heads, seq_len, head_dim)
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1)  # (b, heads, seq_len, seq_len)
        scores.masked_fill_(self.mask[:seq_len, :seq_len], float("-inf"))

        attn_weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = attn_weights @ V  # (b, heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_out)

        # Final projection
        return self.out_proj(context)


def main() -> None:
    torch.manual_seed(123)

    # Define the tensor with 3 rows and 6 columns
    inputs = torch.tensor([
        [0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
        [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
        [0.77, 0.25, 0.10, 0.05, 0.80, 0.55],  # Row 3
    ])

    # Batch the inputs
    batch = torch.stack((inputs, inputs), dim=0)
    print("batch.shape:", batch.shape)

    batch_size, context_length, d_in = batch.shape
    d_out = 6
    num_heads = 2

    # Instantiate and run MultiHeadAttention
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)
    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    main()
