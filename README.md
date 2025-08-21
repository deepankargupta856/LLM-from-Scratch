# GPT-2 from Scratch (124M)

A from-scratch implementation of a GPT-2 style autoregressive transformer in PyTorch. Implements tokenizer, positional encodings, multi-head causal self-attention, MLP, training loop, evaluation, and sampling — all without external transformer libraries.

---

## Quick summary

* **Model:** GPT-2 style decoder-only transformer
* **Model size:** 124,412,160 parameters
* **Training compute:** NVIDIA 3050RTX (local machine)
* **Model checkpoint:** [https://huggingface.co/DukkiZamindar/Self\_Built\_LLM](https://huggingface.co/DukkiZamindar/Self_Built_LLM)
* **Goal:** Architectural fidelity, diagnostics, and reproducibility on limited compute

---

## Model configuration

| Parameter           | Value                  |
| ------------------- | ---------------------- |
| Vocabulary size     | 50,257                 |
| Context length      | 1,024 tokens           |
| Embedding dimension | 768                    |
| Attention heads     | 12                     |
| Transformer layers  | 12                     |
| Dropout rate        | 0.1                    |
| QKV bias            | False                  |
| Model size          | 124,412,160 parameters |

---

## Training details

|                Detail |                      Value |
| --------------------: | -------------------------: |
|           Tokens seen |              \~1.1 million |
|                Epochs |                          2 |
|         Training time | 59.03 minutes (on 3050RTX) |
|   Final training loss |                      5.416 |
| Final validation loss |                      5.330 |
|         Eval log loss |                    11.0064 |
|  Inference throughput |           24.00 tokens/sec |

---

## Evaluation notes

* Evaluation performed on a held-out validation set using negative log likelihood.
* High perplexity is expected given the small corpus and limited training steps.
* Focus is on architecture, diagnostics, and reproducibility rather than benchmark performance.

---

## Key repository features

* Custom tokenizer (tiktoken / BPE-like configuration)
* Complete transformer stack: embeddings, positional encodings, multi-head causal attention, MLPs
* Training loop with logging of loss, perplexity, and tokens/sec
* Evaluation script computing negative log loss and perplexity
* Sampling code (greedy / top-k / top-p where applicable)
* Utilities: attention visualization, gradient norms, basic LR scheduling hooks

---

## Limitations & future work

* Loss stagnation observed after \~1.1M tokens — scale dataset to 10M+ tokens and increase training steps.
* Add learning rate warmup + cosine decay scheduler.
* Add advanced regularization: LayerDrop, label smoothing.
* Improve sampling (top-k / nucleus) and repetition mitigation.
* Profile and optimize memory/compute for larger batch sizes (AMP + gradient accumulation).

---

## How to reproduce

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Prepare dataset:

```bash
python scripts/build_dataset.py --input raw_text_folder --out data/tokenized_dataset.pt
```

3. Train:

```bash
python train.py --config configs/gpt_config_124m.json --device cuda --epochs 2
```

4. Evaluate:

```bash
python evaluate.py --checkpoint checkpoints/latest.pt --data data/val.pt
```

5. Sample:

```bash
python sample.py --checkpoint checkpoints/latest.pt --prompt "He said,"
```

---

