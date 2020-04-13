import os
import shutil
import re
import argparse
import jieba
import pickle
from collections import defaultdict
import numpy as np
import torch


class TextDataset:
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        x = self.arr[item, :]

        y = torch.zeros(x.shape)
        y[:-1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.arr.shape[0]


class Corpus:
    def __init__(self, text_path=None, max_vocab=5000):

        if isinstance(text_path, dict):
            self.text = text_path['text']
            self.vocab = text_path['vocab']
            self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
            self.int_to_word_table = dict(enumerate(self.vocab))
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

            text = re.sub(r'[\r\n\s]+', ' ', text)
            
            self.text = text

            vocab = list(set(text))
            vocab_count = defaultdict(int)

            if len(vocab) > max_vocab:
                for word in text:
                    vocab_count[word] += 1
                vocab_count_list = []
                for word in vocab_count:
                    vocab_count_list.append((word, vocab_count[word]))
                vocab_count_list.sort(key=lambda x: x[1], reverse=True)
                vocab_count_list = vocab_count_list[:max_vocab]
                vocab = [x[0] for x in vocab_count_list]

            self.vocab = vocab
            self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
            self.int_to_word_table = dict(enumerate(self.vocab))

    def save(self, pth):
        torch.save({
            'text': self.text,
            'vocab': self.vocab
        }, pth)

    @property
    def length(self):
        return len(self.text)

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '?'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Index error')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)
