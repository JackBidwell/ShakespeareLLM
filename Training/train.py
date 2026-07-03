import math
import os

import torch

from Training.tokenizer import Tokenizer
from Training.dataset import build_splits, get_batch
from Models.Config import GPTConfig
from Models.Transformer import GPTModel

batch_size = 64
max_iters = 6000
eval_interval = 250
eval_iters = 100
warmup_iters = 200
learning_rate = 3e-4
min_lr = learning_rate / 10
weight_decay = 0.1
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer("Data/Raw/shakespeare_clean.txt")
train_data, val_data = build_splits(tokenizer, val_fraction=0.1)

config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPTModel(config).to(device)

if os.path.exists("checkpoint.pt"):
    try:
        model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
        print("Loaded existing checkpoint.pt")
    except RuntimeError as e:
        print(f"checkpoint.pt is incompatible with the current model config, starting fresh ({e})")

decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95),
)


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
            x, y = get_batch(data, config.block_size, batch_size, device)
            logits = model(x)
            B, T, V = logits.shape
            losses[i] = torch.nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T)).item()
        out[name] = losses.mean().item()
    model.train()
    return out


model.train()
best_val_loss = float("inf")

for it in range(max_iters):
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch(train_data, config.block_size, batch_size, device)

    logits = model(x)
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"Iter {it} | LR {lr:.2e} | Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")

        torch.save(model.state_dict(), "checkpoint.pt")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), "checkpoint_best.pt")
            print(f"New best val loss {best_val_loss:.4f}, saved checkpoint_best.pt")

print("Training complete.")
