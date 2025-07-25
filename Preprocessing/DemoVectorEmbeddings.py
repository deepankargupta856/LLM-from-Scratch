import torch

# Sample input tensor of token indices
input_ids = torch.tensor([2, 3, 5, 1])

# Vocabulary size and embedding output dimension
vocab_size = 6
output_dim = 3

# Set seed for reproducibility
torch.manual_seed(123)

# Create embedding layer
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Print embedding weights
print(embedding_layer.weight)

# Get embedding for a single token
print(embedding_layer(torch.tensor([3])))

# Get embeddings for full input sequence
print(embedding_layer(input_ids))
