import pytest

import model.tokenizer as tkn

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barked at the fox",
    "the fox ran quickly away",
]


def test_vocab():
    tokenizer = tkn.BPETokenizer(vocab_size=100)
    tokenizer.train(CORPUS)
    # test properties that must hold regardless of merge order
    assert len(tokenizer.rev_vocab) <= 100
    assert "<unk>" in tokenizer.rev_vocab
    assert "<pad>" in tokenizer.rev_vocab
    # all base characters from corpus should be present
    for char in set("".join(CORPUS)):
        if char != " ":
            assert char in tokenizer.rev_vocab


def test_encode():
    tokenizer = tkn.BPETokenizer(vocab_size=100)
    tokenizer.train(CORPUS)
    ids = tokenizer.encode("the fox jumps")
    # check types and that all ids are valid
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(i in tokenizer.vocabulary for i in ids)


def test_decode():
    tokenizer = tkn.BPETokenizer(vocab_size=100)
    tokenizer.train(CORPUS)
    text = "the fox jumps"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_decode_corpus():
    # round-trip every sentence in training corpus
    tokenizer = tkn.BPETokenizer(vocab_size=100)
    tokenizer.train(CORPUS)
    for sentence in CORPUS:
        assert tokenizer.decode(tokenizer.encode(sentence)) == sentence
