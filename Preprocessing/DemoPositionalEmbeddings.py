import torch
from dataloader import create_dataloader_v1

# Define embedding parameters
vocab_size = 50257       # GPT-2 vocab size
embedding_dim = 256      # Output dimension of embeddings
context_length = 4       # Length of token sequence per input

# Initialize token embedding layer
torch.manual_seed(42)    # Ensures reproducible embedding initialization
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

# Load input text
with open("../resources/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Create dataloader with non-overlapping sequences
dataloader = create_dataloader_v1(
    txt=raw_text,
    batch_size=8,
    max_length=context_length,
    stride=context_length,  # Non-overlapping chunks
    shuffle=False
)

# Fetch a batch of token IDs
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Apply token embedding
token_embeddings = token_embedding_layer(inputs)
print("Token embeddings shape:", token_embeddings.shape)

# Initialize and apply positional embedding
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)
positions = torch.arange(context_length).unsqueeze(0).expand(inputs.size(0), -1)
pos_embeddings = pos_embedding_layer(positions)
print("Positional embeddings shape:", pos_embeddings.shape)

# Combine token and positional embeddings
input_embeddings = token_embeddings + pos_embeddings
print("Final input embeddings shape:", input_embeddings.shape)
