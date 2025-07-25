import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # Split on punctuation, double-dash, or whitespace
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unknown tokens
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        # Convert tokens to IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # Convert IDs back to tokens and join
        text = " ".join(self.int_to_str[i] for i in ids)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# Load and preprocess the raw text
with open("../resources/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

# Build vocabulary
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: idx for idx, token in enumerate(all_tokens)}

print(len(vocab))
for token, idx in list(vocab.items())[-5:]:
    print(token, idx)

# Initialize tokenizer and test
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print(text)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
