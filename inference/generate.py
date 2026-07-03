import torch
import torch.nn.functional as F

from Training.tokenizer import Tokenizer
from Models.Config import GPTConfig
from Models.Transformer import GPTModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer("Data/Raw/shakespeare_clean.txt")

config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPTModel(config).to(device)

model.load_state_dict(torch.load("checkpoint_best.pt", map_location=device))
model.eval()


def sample_next(logits, temperature=0.8, top_k=40):
    logits = logits / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -model.block_size:]
        logits = model(tokens_cond)
        logits = logits[:, -1, :]
        next_token = sample_next(logits, temperature, top_k)
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
            temperature=0.8,
            top_k=40,
        )

        print("\n--- OUTPUT ---\n")
        print(output)
        print("\n--------------\n")
