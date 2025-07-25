import importlib
import tiktoken

# Initialize tokenizer for GPT-2
tokenizer = tiktoken.get_encoding("gpt2")

# Sample text
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

# Encode the text to token IDs
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Encoded token IDs:", integers)

# Decode the token IDs back to string
strings = tokenizer.decode(integers)
print("Decoded string:", strings)

# Define multiple tokenizers
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
    "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
}

# Get the vocabulary size for each encoding
vocab_sizes = {model: encoding.n_vocab for model, encoding in encodings.items()}

# Print the vocabulary sizes
for model, size in vocab_sizes.items():
    print(f"The vocabulary size for {model.upper()} is: {size}")
