"""
Measures how much of a generated text is copied verbatim from the training
corpus, as n-gram overlap. High overlap on a small fine-tuning corpus means
the model is quoting rather than composing.

Usage:
    python -m tools.check_memorization "generated text here" [--n 8]
"""

import argparse

CORPUS_PATH = "Data/Raw/shakespeare_clean.txt"


def word_ngrams(text, n):
    words = text.split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlap_ratio(generated, corpus_text, n=8):
    gen_grams = word_ngrams(generated, n)
    if not gen_grams:
        return 0.0, 0

    corpus_grams = word_ngrams(corpus_text, n)
    copied = gen_grams & corpus_grams
    return len(copied) / len(gen_grams), len(copied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Generated text to check")
    parser.add_argument("--n", type=int, default=8, help="N-gram size (default: 8 words)")
    args = parser.parse_args()

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    ratio, copied_count = overlap_ratio(args.text, corpus_text, args.n)

    print(f"{args.n}-word n-grams also found verbatim in the training corpus: "
          f"{ratio:.1%} ({copied_count} n-grams)")

    if ratio > 0.5:
        print("-> Likely quoting the corpus rather than composing.")
    elif ratio > 0.15:
        print("-> Some verbatim phrases, but mostly novel construction.")
    else:
        print("-> Mostly novel construction.")


if __name__ == "__main__":
    main()
