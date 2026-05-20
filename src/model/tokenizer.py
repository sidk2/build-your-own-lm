"""
Implementation of a byte-pair encoding tokenizer.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

PRETOK_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+""")


class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
        self.vocabulary: Dict[int, str] = {}
        self.rev_vocab: Dict[str, int] = {}
        self.token_ids: Dict[Tuple[int, int], int] = {}
        self.merge_rank: Dict[Tuple[int, int], int] = {}

    def pretokenize(self, text: str) -> List[str]:
        return re.findall(PRETOK_PATTERN, text)

    def get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        vocab = defaultdict(int)
        for text in corpus:
            for word in self.pretokenize(text):
                word = word.strip()
                if word:
                    vocab[" ".join(list(word)) + " </w>"] += 1
        return vocab

    def get_pair_stats(self, vocab: Dict[str, int]) -> Dict[Tuple, int]:
        pair_frequency: Dict[Tuple, int] = defaultdict(int)
        for item, freq in vocab.items():
            toks = item.split()
            for idx in range(len(toks) - 1):
                pair_frequency[(toks[idx], toks[idx + 1])] += freq
        return pair_frequency

    def merge_vocab(
        self, vocab: Dict[str, int], pair_stats: Dict[Tuple, int]
    ) -> Tuple[Dict[str, int], Tuple]:
        best_pair = max(pair_stats, key=pair_stats.get)
        new_vocab = {}
        for token, freq in vocab.items():
            tok_list = token.split()
            new_tok_list = []
            idx = 0
            while idx < len(tok_list):
                if (
                    idx < len(tok_list) - 1
                    and (tok_list[idx], tok_list[idx + 1]) == best_pair
                ):
                    new_tok_list.append(tok_list[idx] + tok_list[idx + 1])
                    idx += 2
                else:
                    new_tok_list.append(tok_list[idx])
                    idx += 1
            new_vocab[" ".join(new_tok_list)] = freq
        return new_vocab, best_pair

    def train_bpe(
        self, corpus: List[str]
    ) -> Tuple[Dict[str, int], List[Tuple[str, str]], Dict[str, int]]:
        merge_rules = []
        vocab = self.get_vocab(corpus)
        initial_vocab = vocab.copy()  # snapshot before any merges
        initial_vocab_size = len(set(sym for word in vocab for sym in word.split()))
        num_merges = self.vocab_size - initial_vocab_size
        for _ in range(num_merges):
            stats = self.get_pair_stats(vocab)
            if not stats:
                break
            vocab, merge_rule = self.merge_vocab(vocab, stats)
            merge_rules.append(merge_rule)
        return vocab, merge_rules, initial_vocab

    def build_token_vocab(
        self,
        vocab: Dict[str, int],
        initial_vocab: Dict[str, int] = None,
        merge_rules: List[Tuple] = None,
    ) -> Dict[str, int]:
        tokens = set()
        for word in vocab:
            tokens.update(word.split())
        if initial_vocab:
            for word in initial_vocab:
                tokens.update(word.split())
        if merge_rules:
            for a, b in merge_rules:
                tokens.add(a + b)  # every intermediate merged token
        tokens.update(["<unk>", "<pad>"])
        return {tok: idx for idx, tok in enumerate(sorted(tokens))}

    def train(self, corpus: List[str]):
        vocab, merge_rules, initial_vocab = self.train_bpe(corpus)
        token2id = self.build_token_vocab(vocab, initial_vocab, merge_rules)

        self.rev_vocab = token2id
        self.vocabulary = {v: k for k, v in token2id.items()}

        for rank, (a, b) in enumerate(merge_rules):
            id_a = token2id[a]
            id_b = token2id[b]
            merged_id = token2id[a + b]
            self.token_ids[(id_a, id_b)] = merged_id
            self.merge_rank[(id_a, id_b)] = rank

    def encode(self, text: str) -> List[int]:
        all_ids = []
        for word in self.pretokenize(text):
            word = word.strip()
            if not word:
                continue

            symbols = list(word) + ["</w>"]
            ids = [self.rev_vocab.get(s, self.rev_vocab["<unk>"]) for s in symbols]

            while len(ids) > 1:
                best_rank = float("inf")
                best_idx = None
                for i in range(len(ids) - 1):
                    rank = self.merge_rank.get((ids[i], ids[i + 1]), float("inf"))
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i

                if best_idx is None:
                    break

                merged_id = self.token_ids[(ids[best_idx], ids[best_idx + 1])]
                ids = ids[:best_idx] + [merged_id] + ids[best_idx + 2 :]

            all_ids.extend(ids)
        return all_ids

    def decode(self, tokens: List[int]) -> str:
        text = "".join(self.vocabulary.get(i, "<unk>") for i in tokens)
        return text.replace("</w>", " ").strip()


if __name__ == "__main__":
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barked at the fox",
        "the fox ran quickly away",
    ]

    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)

    print(tokenizer.rev_vocab)

    text = "the fox jumps"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"text:    {text}")
    print(f"encoded: {ids}")
    print(f"decoded: {decoded}")
    print(f"vocab size: {len(tokenizer.vocabulary)}")
