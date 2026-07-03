import os

from tokenizers import ByteLevelBPETokenizer

DATA_PATH = "Data/Raw/shakespeare_clean.txt"
OUT_DIR = "Data/Processed/tokenizer"
VOCAB_SIZE = 8000

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[DATA_PATH],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
    )
    tokenizer.save_model(OUT_DIR)
    print(f"Saved BPE tokenizer ({VOCAB_SIZE} vocab) to {OUT_DIR}")
