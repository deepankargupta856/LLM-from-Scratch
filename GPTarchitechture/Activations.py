#!/usr/bin/env python3
"""
This module implements the GELU activation function and compares it
to ReLU by plotting both over a range of input values.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    Approximation as used in the GPT‑2 paper:
        GELU(x) = 0.5 * x * [1 + tanh( sqrt(2/π) * (x + 0.044715 x^3) )]
    """
    def __init__(self):
        """Initialize the GELU module (no parameters)."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the GELU activation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: GELU-activated output tensor.
        """
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * x.pow(3))
            )
        )


def plot_activations():
    """
    Generate a side-by-side plot comparing GELU and ReLU activations
    over the range [-3, 3].
    """
    # Instantiate activation functions
    gelu = GELU()
    relu = nn.ReLU()

    # Sample input values
    x = torch.linspace(-3, 3, 100)

    # Compute activations
    y_gelu = gelu(x)
    y_relu = relu(x)

    # Configure the figure
    plt.figure(figsize=(8, 3))

    # Plot GELU and ReLU in separate subplots
    for idx, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), start=1):
        plt.subplot(1, 2, idx)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Entry point: plot and compare activation functions."""
    plot_activations()


if __name__ == "__main__":
    main()
