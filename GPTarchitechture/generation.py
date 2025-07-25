import torch
import tiktoken
from GPTarchitechture.GPTmodel import MyGPT


# -----------------------------#
# Configuration
# -----------------------------#
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


# -----------------------------#
# Text Generation Utilities
# -----------------------------#
class Generate:
    @staticmethod
    def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
        """
        Encode text into a tensor of token IDs.
        """
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        return torch.tensor(encoded).unsqueeze(0)

    @staticmethod
    def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
        """
        Decode a tensor of token IDs back into text.
        """
        return tokenizer.decode(token_ids.squeeze(0).tolist())

    @staticmethod
    def generate_text_simple(
        model: torch.nn.Module,
        idx: torch.Tensor,
        max_new_tokens: int,
        context_size: int
    ) -> torch.Tensor:
        """
        Generate tokens from a model given an initial context.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)

            logits = logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
    @staticmethod
    def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx

# -----------------------------#
# Main Generation Logic
# -----------------------------#
def main():
    start_prompt = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    input_ids = Generate.text_to_token_ids(start_prompt, tokenizer)

    model = MyGPT(GPT_CONFIG_124M)
    model.eval()

    generated_ids = Generate.generate_text_simple(
        model=model,
        idx=input_ids,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    generated_text = Generate.token_ids_to_text(generated_ids, tokenizer)
    print("Generated Text:\n", generated_text)


if __name__ == "__main__":
    main()
