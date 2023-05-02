from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math


class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len,dropout=0.2):
        # d_model is number of dimension of the embedding vector
        super(EmbeddingWithPositionalEncoding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x has shape (batch_size, seq_len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.positional_embedding(pos)
        token_emb = self.token_embedding(x)  # shape (batch_size, seq_len, d_model)
        return self.dropout(token_emb + pos_emb)

class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size,block_size,dropout=0.2):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, _input):
        B,T,C = _input.shape
        k = self.key(_input)
        q = self.query(_input)
        wei = q @ k.transpose(-2, -1)

        # tril = torch.tril(torch.ones(_input.size(1), _input.size(1)))
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(_input)
        out = wei @ v
        return out
    
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size,embed_size,block_size) -> None:
        super().__init__()
        """multiple attention heads in parallel"""

        """Args:
                num_heads:number of Attention heads that used in MultiHeadAttention
                head_size:output dim of key and query
                embed_size:Embedding dimension
        """
        self.heads = nn.ModuleList([
            AttentionHead(embed_size=embed_size,head_size=head_size,block_size=block_size)
            for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size,embed_size) # projection layer followed by MHA

    def forward(self,X):
        out = torch.cat([h(X) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embed,dropout=0.2):
        super().__init__()
        """Simple Position-wise FeedForward Neural Network for our transformer block"""
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed*4),
            nn.GELU(),
            nn.Linear(n_embed*4,n_embed),
            nn.Dropout(dropout)
        )

    def forward(self,X):
        out = self.net(X)
        return out
    
class AddNorm(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embed)

    def forward(self, x, sub_layer_output):
        # Add and normalize
        # LayerNorm(x + Sublayer(x))
        output = self.layer_norm(x + sub_layer_output)
        return output
    
class Block(nn.Module):
    def __init__(self,d_model,max_seq_len,n_embed,n_heads,dropout=0.2):
        super().__init__()
        head_size = n_embed // n_heads
        self.masked_attention = MaskedMultiHeadAttention(
            num_heads=n_heads,
            head_size=head_size,
            embed_size=n_embed,
            block_size=max_seq_len
        )
        
        self.ln1 = AddNorm(n_embed=n_embed)
        self.ln2 = AddNorm(n_embed=n_embed)
        self.feedforward = FeedForward(
            n_embed=n_embed,
            dropout=0.1
        )
        self.ln3 = AddNorm(n_embed=n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,embed_output):
        identity = embed_output
        masked_attn_out = self.masked_attention(embed_output)
        out = self.ln1(masked_attn_out,identity)
        identity = out
        out = self.feedforward(out)               
        out = self.ln2(out,identity)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 n_heads,
                 num_layers,
                 max_seq_len=40
            ):
        super().__init__()
        self.embedding = EmbeddingWithPositionalEncoding(vocab_size=vocab_size,
                                                               d_model=embed_dim,
                                                               max_seq_len=max_seq_len)

        
        self.blocks = nn.ModuleList(
            [Block(d_model=embed_dim,
                          max_seq_len=max_seq_len,
                          n_embed=embed_dim,
                          n_heads=n_heads)]
            )

        self.fc = nn.Linear(embed_dim,vocab_size)
    
    def forward(self,inp):
        # Embed question and answer sequences
        embed = self.embedding(inp)
        # # Apply Decoder layers
        attention_mask = self._generate_square_subsequent_mask(inp)
        # output,_ = self.rnn(embed)
        output = embed
        for layer in self.blocks:
            output = layer(output)
        
        # output,_ = self.lstm(output)
        # # Map decoder output to answer vocabulary using linear layer
        output = self.fc(output)
        # output = F.softmax(logits)
        return output
    
    def _generate_square_subsequent_mask(self, tensor):
        mask = (torch.triu(torch.ones(tensor.size(-1), tensor.size(-1))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(tensor.device)



# To model a RNN with attention using PyTorch, you can use the torch.nn module to define the encoder and decoder networks, and implement a custom attention layer using torch.bmm to perform batch matrix multiplication. Here is an example of a RNN with attention based on the paper “Neural Machine Translation by Jointly Learning to Align and Translate” by Bahdanau et al. (2015):

