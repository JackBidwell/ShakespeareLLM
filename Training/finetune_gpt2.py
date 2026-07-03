import math
import os

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CORPUS_PATH = "Data/Raw/shakespeare_clean.txt"
OUT_DIR = "Models/gpt2_shakespeare"

batch_size = 8
block_size = 512
max_iters = 3000
eval_interval = 100
eval_iters = 50
warmup_iters = 100
learning_rate = 5e-5  # small: fine-tuning, not pretraining
min_lr = learning_rate / 10
weight_decay = 0.1
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)


def build_splits(val_fraction=0.1):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer(text)["input_ids"]
    data = torch.tensor(ids, dtype=torch.long)
    split = int(len(data) * (1 - val_fraction))
    return data[:split], data[split:]


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


train_data, val_data = build_splits()
print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95),
)

scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with torch.amp.autocast(device, enabled=(device == "cuda"), dtype=torch.float16):
                logits = model(x).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


model.train()
best_val_loss = float("inf")

for it in range(max_iters):
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size, device)

    with torch.amp.autocast(device, enabled=(device == "cuda"), dtype=torch.float16):
        logits = model(x).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"Iter {it} | LR {lr:.2e} | Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")

        model.save_pretrained(f"{OUT_DIR}/last")
        tokenizer.save_pretrained(f"{OUT_DIR}/last")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            model.save_pretrained(f"{OUT_DIR}/best")
            tokenizer.save_pretrained(f"{OUT_DIR}/best")
            print(f"New best val loss {best_val_loss:.4f}, saved to {OUT_DIR}/best")

print("Fine-tuning complete.")
