import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from GPTarchitechture.GPTmodel import MyGPT, GPT_CONFIG_124M


def calculate_loss_and_perplexity(model, dataset, tokenizer, device='cuda', batch_size=1, max_length=1024):
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Tokenize and prepare inputs
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            labels = input_ids.clone()

            logits = model(input_ids)  # shape: [B, T, V]

            shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
            shift_labels = input_ids[:, 1:].contiguous()  # [B, T-1]
            shift_mask = attention_mask[:, 1:]  # [B, T-1]

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )

            # Mask the padding tokens
            loss = loss.view(shift_labels.shape)
            masked_loss = (loss * shift_mask).sum()

            num_tokens = shift_mask.sum().item()
            total_loss += masked_loss.item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"Average Negative Log Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity.item()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    from datasets import load_dataset

    # Change these as needed
    model_name = "gpt2"
    dataset_name = "wikitext"
    dataset_split = "test"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = MyGPT(GPT_CONFIG_124M)

    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=dataset_split)
    texts = dataset["text"]
    texts = [t for t in texts if len(t.strip()) > 0]  # Remove empty lines

    calculate_loss_and_perplexity(model, texts, tokenizer)
