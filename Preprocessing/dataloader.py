import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=False, drop_last=True,
                         num_workers=0):
    """
    Creates a DataLoader with tokenized input-target pairs using a sliding window.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

if __name__ == "__main__":
    # --------------------------------------------------------------
    #   What does "if __name__ == '__main__':" mean?
    # --------------------------------------------------------------
    # - This block ensures the code runs only when the script is executed directly.
    # - If this script is imported as a module in another script,
    #   everything inside this block will NOT run automatically.
    # - This is useful for testing or running standalone scripts without
    #   triggering logic during import.
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    #   How stride and batching interact in tokenized datasets
    # --------------------------------------------------------------
    # - `stride` controls how far the sliding window moves to create the next
    #   input-target pair in the dataset. It is applied globally.
    # - Each training sample is a fixed-length sequence of tokens (`max_length`).
    # - The next sample starts `stride` tokens after the start of the previous sample.
    # - Batches are formed by grouping these samples. `batch_size` does not
    #   influence how stride or the windowing logic works.
    # - Therefore, samples in the same batch are simply adjacent entries
    #   in the dataset list, each already spaced by the given `stride`.
    # --------------------------------------------------------------

    # Load input text
    with open("../resources/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("PyTorch version:", torch.__version__)

    # Dataloader with batch size 1
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=2, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("First batch:", first_batch)

    second_batch = next(data_iter)
    print("Second batch:", second_batch)

    # Dataloader with batch size 8
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=2, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("\nInputs (batch 1):\n", inputs)
    print("\nTargets (batch 1):\n", targets)

    next_inputs, next_targets = next(data_iter)
    print("\nInputs (batch 2):\n", next_inputs)
    print("\nTargets (batch 2):\n", next_targets)
