"""
Parses a training log (stdout from Training/train.py or
Training/finetune_gpt2.py, saved to a file) and plots train/val loss
over iterations.

Usage:
    python -m tools.plot_loss path/to/log.txt --out loss.png
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"Iter (\d+) \| LR [\d.e+-]+ \| Train Loss ([\d.]+) \| Val Loss ([\d.]+)"
)


def parse_log(path):
    iters, train_losses, val_losses = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                iters.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    return iters, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to a saved training log")
    parser.add_argument("--out", default="loss.png", help="Output image path")
    args = parser.parse_args()

    iters, train_losses, val_losses = parse_log(args.log_path)

    if not iters:
        raise SystemExit(f"No 'Iter ... Loss ...' lines found in {args.log_path}")

    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])

    plt.figure(figsize=(8, 5))
    plt.plot(iters, train_losses, label="train loss")
    plt.plot(iters, val_losses, label="val loss")
    plt.scatter([iters[best_idx]], [val_losses[best_idx]], color="red", zorder=5,
                label=f"best val ({val_losses[best_idx]:.3f} @ iter {iters[best_idx]})")
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title(args.log_path)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    print(f"Best val loss: {val_losses[best_idx]:.4f} at iter {iters[best_idx]}")


if __name__ == "__main__":
    main()
