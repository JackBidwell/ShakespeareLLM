from tools.check_memorization import overlap_ratio, word_ngrams


def test_word_ngrams_basic():
    grams = word_ngrams("the quick brown fox", 2)

    assert grams == {("the", "quick"), ("quick", "brown"), ("brown", "fox")}


def test_overlap_ratio_full_copy():
    corpus = "to be or not to be that is the question"
    generated = "to be or not to be"

    ratio, count = overlap_ratio(generated, corpus, n=3)

    assert ratio == 1.0
    assert count > 0


def test_overlap_ratio_novel_text():
    corpus = "to be or not to be that is the question"
    generated = "purple elephants dance beneath the shimmering moonlight tonight"

    ratio, count = overlap_ratio(generated, corpus, n=3)

    assert ratio == 0.0
    assert count == 0


def test_overlap_ratio_empty_generated():
    ratio, count = overlap_ratio("", "some corpus text here", n=3)

    assert ratio == 0.0
    assert count == 0
