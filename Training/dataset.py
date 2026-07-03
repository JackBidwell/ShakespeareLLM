import torch


def build_splits(tokenizer, val_fraction=0.1):
    data = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)
    split = int(len(data) * (1 - val_fraction))
    return data[:split], data[split:]


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
