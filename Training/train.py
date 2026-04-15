import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.tokenizer import Tokenizer
from training.dataset import ShakespeareDataset
from models.transformer import GPTModel

batch_size = 32
block_size = 256
epochs = 3
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
max_batches_per_epoch = 1000

tokenizer = Tokenizer("data/raw/shakespeare.txt")

dataset = ShakespeareDataset(
    tokenizer=tokenizer,
    block_size=block_size
)

vocab_size = tokenizer.vocab_size

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = GPTModel(
    vocab_size=vocab_size,
    embed_size=128,
    num_layers=4,
    num_heads=4,
    block_size=block_size
).to(device)

model.train()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    batches_run = 0

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches_per_epoch:
            print(f"Stopping epoch {epoch} at batch {batch_idx}")
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        y = y.view(B * T)

        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        batches_run += 1

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

    avg_loss = total_loss / batches_run
    print(f"Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("Model saved to checkpoint.pt")

print("Training complete.")