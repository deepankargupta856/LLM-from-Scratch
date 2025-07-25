"""
A minimal self-attention example with 3D visualization of word embeddings
and a toy `SimpleSelfAttention` module.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# its more advisable to use softmax to normalize due to softmax being better at managing
# extreme values and offers more favorable gradient properties during training
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

def stable_softmax(x, dim=0):
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    return e_x / e_x.sum(dim=dim, keepdim=True)

class SimpleSelfAttention(nn.Module):
    def forward(self, x):
        attn_scores = x @ x.T
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = attn_weights @ x
        return context_vec

if __name__ == "__main__":
    # 1) Prepare toy embeddings and labels
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89],  # Your
       [0.55, 0.87, 0.66],  # journey
       [0.57, 0.85, 0.64],  # starts
       [0.22, 0.58, 0.33],  # with
       [0.77, 0.25, 0.10],  # one
       [0.05, 0.80, 0.55]]  # step
    )

    query = inputs[1]
    attn_scores = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores[i] = torch.dot(x_i, query)

    print()
    attn_weights = attn_scores / attn_scores.sum()
    print("Attention scores :", attn_scores)
    print("Attention weights :", attn_weights)
    print("Sum:", attn_weights.sum())

    attn_weights_naive = softmax_naive(attn_scores)
    attn_weights_stable = stable_softmax(attn_scores)
    print("Attention weights (Naive) :", attn_weights_naive)
    print("Sum:", attn_weights_naive.sum())
    print("Attention weights (Stable) :", attn_weights_stable)
    print("Sum:", attn_weights_stable.sum())

    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_stable[i] * x_i
    print("Context vector:", context_vec_2)

    # Corresponding words
    words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

    # Extract x, y, z coordinates
    x_coords = inputs[:, 0].numpy()
    y_coords = inputs[:, 1].numpy()
    z_coords = inputs[:, 2].numpy()

    # 2) First 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z, word in zip(x_coords, y_coords, z_coords, words):
        ax.scatter(x, y, z)
        ax.text(x, y, z, word, fontsize=10)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.title('3D Plot of Word Embeddings')
    plt.show()

    # 3) 3D vector plot + context
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
        ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
        ax.text(x, y, z, word, fontsize=10, color=color)
    context_np = context_vec_2.numpy()
    ax.quiver(0, 0, 0, *context_np, color='red', arrow_length_ratio=0.05)
    ax.scatter(*context_np, color='red', s=60)
    ax.text(*context_np, "context_vec", fontsize=10, color='black')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.title('3D Plot of Word Embeddings with Colored Vectors')
    plt.show()

    # 4) Full attention matrix & contexts
    attn_scores_full = inputs @ inputs.T
    print("ATTENTION SCORES FULL:\n", attn_scores_full)
    attn_weights_full = stable_softmax(attn_scores_full, dim=-1)
    print("ATTENTION WEIGHTS FULL:\n", attn_weights_full)
    print("ALL ROW SUMS:\n", attn_weights_full.sum(dim=-1))
    context_vecs = attn_weights_full @ inputs
    print("CONTEXT VECTORS:\n", context_vecs)

    # 5) Validate with nn.Module
    attention = SimpleSelfAttention()
    print("CONTEXT VECS FROM CLASS:\n", attention(inputs))
