from Training.tokenizer import Tokenizer

CORPUS_PATH = "Data/Raw/shakespeare_clean.txt"


def test_encode_decode_roundtrip():
    tokenizer = Tokenizer(CORPUS_PATH)
    text = "O, what light through yonder window breaks?"

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded.strip() == text


def test_vocab_size_matches_tokenizer():
    tokenizer = Tokenizer(CORPUS_PATH)

    assert tokenizer.vocab_size == tokenizer._tok.get_vocab_size()
    assert tokenizer.vocab_size > 0
