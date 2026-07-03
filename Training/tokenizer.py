from tokenizers import ByteLevelBPETokenizer

TOKENIZER_DIR = "Data/Processed/tokenizer"


class Tokenizer:
    def __init__(self, file_path, tokenizer_dir=TOKENIZER_DIR):
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self._tok = ByteLevelBPETokenizer(
            f"{tokenizer_dir}/vocab.json",
            f"{tokenizer_dir}/merges.txt",
        )
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, tokens):
        return self._tok.decode(tokens)
