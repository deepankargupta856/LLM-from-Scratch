import importlib
import tiktoken
from Tokenizer import tokenizer

# Load raw text
with open("../resources/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Initialize BPE tokenizer
bpe_tokenizer = tiktoken.get_encoding("gpt2")

# Encode using both tokenizers
enc_text = tokenizer.encode(raw_text)
bpe_enc_text = bpe_tokenizer.encode(raw_text)

# Print length of encodings
print("Length of custom encoding:", len(enc_text))
print("Length of BPE encoding:", len(bpe_enc_text))

# Create context samples
enc_sample = enc_text[50:]
bpe_enc_sample = bpe_enc_text[50:]
context_size = 4

# Create input-output pairs
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
bpe_x = bpe_enc_sample[:context_size]
bpe_y = bpe_enc_sample[1:context_size + 1]

# Print input-target pairs (custom tokenizer)
print("\nCUSTOM TOKENIZER INPUT TARGET PAIRS")
print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Print input-target pairs (BPE tokenizer)
print('\nBYTE PAIR ENCODING BASED INPUT TARGET PAIRS')
print(f"x: {bpe_x}")
print(f"y: {bpe_y}")

for i in range(1, context_size + 1):
    context = bpe_enc_sample[:i]
    desired = bpe_enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size + 1):
    context = bpe_enc_sample[:i]
    desired = bpe_enc_sample[i]
    print(bpe_tokenizer.decode(context), "---->", bpe_tokenizer.decode([desired]))

# Tokenizer vocabulary sizes
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
    "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
}

vocab_sizes = {model: encoding.n_vocab for model, encoding in encodings.items()}

print("\nVOCABULARY SIZES")
for model, size in vocab_sizes.items():
    print(f"The vocabulary size for {model.upper()} is: {size}")
