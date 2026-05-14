'''
Implementation of a byte-pair encoding tokenizer.
'''

import os
import torch

from collections import defaultdict

from typing import Dict, List

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size : int = vocab_size
        self.vocabulary : Dict[str, int] = defaultdict(int)


    def train(self, dataset: List[str]):
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
        for snippet in dataset:
            for char in snippet:
                char
                
        while len(self.vocabulary) < self.vocab_size:

        

    def encode(self, text):
        pass

    def decode(self, tokens):
        pass
