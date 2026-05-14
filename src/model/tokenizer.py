'''
Implementation of a byte-pair encoding tokenizer.
'''

import os
import torch

from collections import defaultdict

from typing import Dict, List, Set, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size : int = vocab_size
        self.vocabulary : Dict[int, str] = {}
        self.rev_vocab : Dict[str, int] = {} # maybe use defaultdicts?

        # Maps pair of tokens, (token_id, token_id), to merged id
        self.token_ids : Dict[Tuple[int, int], int] = {}

    def train(self, dataset: List[str], 
                    vocab_size: int,
                    special_tokens : Set[str] = {"|<eot>|"}):
        '''
        Train the tokenizer.

        The BPE algorithm works as follows:
        1. Start with the base vocabulary of all bytes.
        2. Count the frequency of all pairs of tokens in the dataset.
        3. Merge the most frequent pair of tokens.
        4. Repeat until the desired vocabulary size is reached.

        Args:
            dataset: A dataset of text to train the tokenizer on. The
                dataset format is expected to be a list of strings
        '''
        # Step 1: Assign all unique characters in the corpus
        # to token ids

        for sample in dataset:
            for char in sample:
                if char not in self.rev_vocab:
                    token_id = max(self.rev_vocab.values(), -1) + 1
                    self.rev_vocab[char] = token_id
                    self.vocab[token_id] = char

                
        while len(self.vocabulary) < self.vocab_size:

        

    def encode(self, text):
        pass

    def decode(self, tokens):
        pass
