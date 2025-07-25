import torch
import tiktoken

from GPTarchitechture.GPTmodel import MyGPT
from GPTarchitechture.generation import Generate

# GPT model configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# Initialize and load model
model = MyGPT(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()

print(model)

# Set random seed
torch.manual_seed(123)

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
token_ids = Generate.generate(
    model=model,
    idx=Generate.text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

# Print output
print("Output text:\n", Generate.token_ids_to_text(token_ids, tokenizer))
