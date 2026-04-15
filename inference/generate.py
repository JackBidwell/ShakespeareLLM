import torch
import torch.nn.functional as F

from training.tokenizer import Tokenizer
from models.transformer import GPTModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer("data/raw/shakespeare.txt")

model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    embed_size=128,
    num_layers=4,
    num_heads=4,
    block_size=256
).to(device)

model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
model.eval()


def sample_next(logits, temperature=0.8):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.8):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -model.block_size:]
        logits = model(tokens_cond)
        logits = logits[:, -1, :]
        next_token = sample_next(logits, temperature)
        tokens = torch.cat((tokens, next_token), dim=1)

    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    print("\nShakespeare LLM ready.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=300,
            temperature=0.8
        )

        print("\n--- OUTPUT ---\n")
        print(output)
        print("\n--------------\n")