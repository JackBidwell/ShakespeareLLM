class Tokenizer:
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        chars = sorted(list(set(self.text)))

        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.idx_to_char[i] for i in tokens])