"""
Implementation of a byte-pair encoding tokenizer.
"""

import heapq
import multiprocessing
import re
import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

PRETOK_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+""")
N_PROC = multiprocessing.cpu_count()

def process_text(text: str) -> Dict[str, int]:
    text_counts = defaultdict(int)
    for word in PRETOK_PATTERN.findall(text):
        word = word.strip()
        if word:
            text_counts[word] += 1
    return text_counts
    
class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
        self.vocabulary: Dict[int, str] = {}
        self.rev_vocab: Dict[str, int] = {}
        self.token_ids: Dict[Tuple[int, int], int] = {}
        self.merge_rank: Dict[Tuple[int, int], int] = {}

    def save(self, file_path: str) -> None:
        import json
        import os

        data = {
            "vocab_size": self.vocab_size,
            "vocabulary": {str(k): v for k, v in self.vocabulary.items()},
            "rev_vocab": self.rev_vocab,
            "token_ids": {f"{k[0]},{k[1]}": v for k, v in self.token_ids.items()},
            "merge_rank": {f"{k[0]},{k[1]}": v for k, v in self.merge_rank.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load(self, file_path: str) -> None:
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocabulary = {int(k): v for k, v in data["vocabulary"].items()}
        self.rev_vocab = data["rev_vocab"]

        self.token_ids = {}
        for k, v in data["token_ids"].items():
            parts = k.split(",")
            self.token_ids[(int(parts[0]), int(parts[1]))] = v

        self.merge_rank = {}
        for k, v in data["merge_rank"].items():
            parts = k.split(",")
            self.merge_rank[(int(parts[0]), int(parts[1]))] = v
            
    def pretokenize(self, text: str) -> List[str]:
        return re.findall(PRETOK_PATTERN, text)

    def get_vocab(self, corpus: List[str]) -> Dict[str, Tuple[List[str], int]]:
        counts = defaultdict(int)

        with multiprocessing.Pool(processes=N_PROC) as pool:
            results = pool.imap(process_text, corpus)
            for text_counts in tqdm.tqdm(results, desc="Building Initial Vocab", leave=False):
                for word, freq in text_counts.items():
                    counts[word] += freq
        return {word: (list(word) + ["</w>"], freq) for word, freq in counts.items()}

    def get_pair_stats(self, vocab: Dict[str, Tuple[List[str], int]]) -> Dict[Tuple, int]:
        pair_frequency: Dict[Tuple, int] = defaultdict(int)
        for symbols, freq in vocab.values():
            for idx in range(len(symbols) - 1):
                pair_frequency[(symbols[idx], symbols[idx + 1])] += freq
        return pair_frequency

    def build_heap(self, pair_stats: Dict[Tuple, int]) -> List:
        heap = [(-freq, pair) for pair, freq in pair_stats.items()]
        heapq.heapify(heap)
        return heap

    def build_pair_index(self, vocab):
        """Build inverted index: pair -> set of words containing that pair."""
        pair_index = defaultdict(set)
        for word, (symbols, freq) in vocab.items():
            for i in range(len(symbols) - 1):
                pair_index[(symbols[i], symbols[i + 1])].add(word)
        return pair_index

    def pop_best(self, heap, pair_stats) -> Tuple[Tuple, int]:
        while heap:
            neg_freq, pair = heapq.heappop(heap)
            if pair_stats.get(pair, 0) == -neg_freq:
                return pair, -neg_freq
        return None, 0

    def merge_and_update(self, vocab, pair_stats, heap, best_pair, pair_index):
        a, b = best_pair
        merged = a + b

        # Inverted index lets us do merges faster
        affected_words = list(pair_index.get(best_pair, set()))
        pair_index.pop(best_pair, None)

        for word in affected_words:
            symbols, freq = vocab[word]

            if not any(symbols[i] == a and symbols[i + 1] == b for i in range(len(symbols) - 1)):
                continue

            old_pair_set = set((symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1))

            old_counts: Dict[Tuple, int] = defaultdict(int)
            for i in range(len(symbols) - 1):
                old_counts[(symbols[i], symbols[i + 1])] += 1

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            new_pair_set = set((new_symbols[i], new_symbols[i + 1]) for i in range(len(new_symbols) - 1))

            new_counts: Dict[Tuple, int] = defaultdict(int)
            for i in range(len(new_symbols) - 1):
                new_counts[(new_symbols[i], new_symbols[i + 1])] += 1

            # Update pair frequency stats
            for p in set(old_counts) | set(new_counts):
                delta = (new_counts[p] - old_counts[p]) * freq
                if delta != 0:
                    pair_stats[p] += delta
                    heapq.heappush(heap, (-pair_stats[p], p))

            # Update inverted index
            for p in old_pair_set - new_pair_set:
                if p in pair_index:
                    pair_index[p].discard(word)
                    if not pair_index[p]:
                        del pair_index[p]
            for p in new_pair_set - old_pair_set:
                pair_index[p].add(word)

            vocab[word] = (new_symbols, freq)

        return vocab, pair_stats

    def train_bpe(self, corpus: List[str]):
        vocab = self.get_vocab(corpus)
        initial_vocab = {w: (syms[:], freq) for w, (syms, freq) in vocab.items()}

        pair_stats = self.get_pair_stats(vocab)
        heap = self.build_heap(pair_stats)
        pair_index = self.build_pair_index(vocab)

        initial_vocab_size = len(set(s for syms, _ in vocab.values() for s in syms))
        num_merges = self.vocab_size - initial_vocab_size
        merge_rules = []

        for _ in tqdm.trange(num_merges, desc="Training BPE"):
            best_pair, freq = self.pop_best(heap, pair_stats)
            if best_pair is None or freq == 0:
                break
            vocab, pair_stats = self.merge_and_update(vocab, pair_stats, heap, best_pair, pair_index)
            merge_rules.append(best_pair)

        return vocab, merge_rules, initial_vocab

    def build_token_vocab(self, vocab, initial_vocab=None, merge_rules=None) -> Dict[str, int]:
        tokens = set()
        for symbols, _ in vocab.values():
            tokens.update(symbols)
        if initial_vocab:
            for symbols, _ in initial_vocab.values():
                tokens.update(symbols)
        if merge_rules:
            for a, b in merge_rules:
                tokens.add(a + b)
        tokens.update(["<unk>", "<pad>", "<eos>"])
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

            if len(ids) <= 1:
                all_ids.extend(ids)
                continue

            # Min-heap + linked list for O(n log n) merging.
            # Each node i holds tokens[i], linked via prev/next arrays.
            # The heap stores (rank, left_pos, right_pos) with lazy deletion.
            n = len(ids)
            tokens = list(ids)
            prev_node = list(range(-1, n - 1))
            next_node = list(range(1, n + 1))
            active = [True] * n

            encode_heap = []
            for i in range(n - 1):
                rank = self.merge_rank.get((tokens[i], tokens[i + 1]), None)
                if rank is not None:
                    heapq.heappush(encode_heap, (rank, i, i + 1))

            while encode_heap:
                rank, left, right = heapq.heappop(encode_heap)

                # Lazy deletion
                if not active[left] or not active[right] or next_node[left] != right:
                    continue
                pair = (tokens[left], tokens[right])
                if self.merge_rank.get(pair, None) != rank:
                    continue

                tokens[left] = self.token_ids[pair]
                active[right] = False
                next_node[left] = next_node[right]
                if next_node[right] < n:
                    prev_node[next_node[right]] = left

                if prev_node[left] >= 0:
                    new_rank = self.merge_rank.get(
                        (tokens[prev_node[left]], tokens[left]), None
                    )
                    if new_rank is not None:
                        heapq.heappush(encode_heap, (new_rank, prev_node[left], left))

                if next_node[left] < n:
                    new_rank = self.merge_rank.get(
                        (tokens[left], tokens[next_node[left]]), None
                    )
                    if new_rank is not None:
                        heapq.heappush(encode_heap, (new_rank, left, next_node[left]))

            # Walk the linked list to collect surviving tokens
            pos = 0
            while pos < n:
                all_ids.append(tokens[pos])
                pos = next_node[pos]

        return all_ids

    def decode(self, tokens: List[int]) -> str:
        text = "".join(self.vocabulary.get(i, "<unk>") for i in tokens)
        return text.replace("</w>", " ").replace("<eos>", "").replace("<pad>", "").strip()


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