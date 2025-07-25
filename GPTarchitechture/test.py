import torch
from GPTarchitechture.GPTmodel import MyGPT
from Preprocessing.dataloader import create_dataloader_v1


class GPTtrainer:
    @staticmethod
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    @staticmethod
    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = GPTtrainer.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches


if __name__ == "__main__":
    # Config
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Load data
    with open("../resources/the-verdict.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Sanity check
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
        break

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        break

    print("Train loader batches:", len(train_loader))
    print("Val loader batches:", len(val_loader))

    train_tokens = sum(input_batch.numel() for input_batch, _ in train_loader)
    val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyGPT(cfg=GPT_CONFIG_124M)
    model.to(device)

    # Loss evaluation
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = GPTtrainer.calc_loss_loader(train_loader, model, device)
        val_loss = GPTtrainer.calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
