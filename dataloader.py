from typing import Any
import os
import spacy
import string
from torch.utils.data import Dataset , DataLoader
from tqdm import tqdm
import json
import pickle
import torch
import random


spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold=5):

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.wtc = {} # word to count
        self.freq_threshold = freq_threshold

        self.encode = None
        self.decode = None

    def tokenizer_eng(self,text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocab(self,file):
        # frequencies = {}
        idx = 4

        # for sentence in sentence_list:
        for word in file:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx+=1
        
            if word not in self.wtc:
                self.wtc[word] = 1
                
            else:
                self.wtc[word] += 1
        
        # self.encode = self.numericalize_tokens
        # self.decode = lambda l:''.join([self.itos[int(i)] for i in l])

    def chars_tokenizer(self):
        chars = list(set(string.printable))
        vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.encode = lambda s:[self.stoi[c] for c in s]
        self.decode = lambda l:''.join([self.itos[int(i)] for i in l])

    def words_tokenizer(self, tokens):
        # print(tokens.split())
        # tokens = tokens.split()
        vocab_size = len(tokens)
        self.build_vocab(file=tokens)
        # self.stoi = {token: i for i, token in enumerate(tokens)}
        # self.itos = {i: token for i, token in enumerate(tokens)}
        
        self.encode = lambda s: [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in s.split()]
        self.decode = lambda l: ' '.join([self.itos[int(i)] for i in l])

        
    def char_to_idx(self,text):
        return self.encode(text)
    
    def idx_to_char(self,idx_array):
        return self.decode(idx_array)


    def numericalize_tokens(self, text):
        tokenized_text = text.split()

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    
    def index2token(self,tensor):
        pass
    
    def __call__(self,file,_type="word"):
        assert _type in ["char","word"]
        if _type == "word":
            self.words_tokenizer(tokens=file)

        else:
            self.chars_tokenizer()

class TextDataset(Dataset):
    def __init__(self, file_path, vocab, chunk_size, _type="char"):
        self.file_path = file_path
        self.vocab = vocab
        self.chunk_size = chunk_size
        self._type = _type

        with open(file_path, 'r', encoding='utf-8') as f:
            if _type == "word":
                text = f.read()
                self.data = text.split()

            elif _type == "char":
                self.data = f.read()
            
            self.vocab(self.data,_type=_type)

    def __len__(self):
        return (len(self.data) - self.chunk_size) // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size + 1
        chunk = self.data[start_idx:end_idx]
        # print(chunk)
        return self.txt2tensor(chunk)

    def txt2tensor(self, chunk):
        if self._type == "char":
            tokens = list(chunk)
        elif self._type == "word":
            # tokens = chunk.split()
            tokens = chunk
            # print(len(tokens))

        # # Pad the tokens to ensure they are of the same length
        if len(tokens) < self.chunk_size + 1:
            padding_len = self.chunk_size + 1 - len(tokens)
            tokens.extend(['<PAD>'] * padding_len)
        # print(len(tokens))
        indices = [self.vocab.encode(c) for c in tokens]
        tensor = torch.tensor(indices).squeeze(1)
        _input = tensor[:-1]
        output = tensor[1:]
        # _input = tokens[:-1]
        # output = tokens[1:]
        # Return the input and output tensors
        return _input, output