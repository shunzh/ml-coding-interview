"""
To best understand the GPT model, we implement its key components using PyTorch.

The structure of GPT mode is
- Embedding: Map the input tokens to embeddings
- Positional Encoding: Add positional information to the embeddings

- A Stack of Decoder Layers: A stack of decoder layers.
  Inside a decoder layer:
    - Multi-Head Attention: Compute the attention between the input embeddings
    - Layer Normalization: Normalize the output of the multi-head attention
    - Feed Forward: A feed forward neural network
    - Layer Normalization: Normalize the output of the feed forward neural network

- Language Model Head: A linear layer that maps the output of the decoder layers to the vocabulary size

TODO check the following implementation.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class GPTModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
            max_len: int,
    ):
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        # decoder layers
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            input_ids: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        # embedding
        x = self.embedding(input_ids) + self.position_embedding(torch.arange(input_ids.size(1), device=input_ids.device))

        # decoder layers
        for decoder in self.decoders:
            x = decoder(x, mask)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout: float,
    ):
        super().__init__()

        # query, key, value projection
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
            self,
            hidden_state: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        # query, key, value projection
        query = self.query_projection(hidden_state)  # (batch_size, seq_len, hidden_size)
        key = self.key_projection(hidden_state)  # (batch_size, seq_len, hidden_size)
        value = self.value_projection(hidden_state)  # (batch_size, seq_len, hidden_size)

        # split heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads) # (batch_size, seq_len, num_heads, hidden_size // num_heads)
        key = key.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads) # (batch_size, seq_len, num_heads, hidden_size // num_heads)
        value = value.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads) # (batch_size, seq_len, num_heads, hidden_size // num_heads)

        # transpose
        query = query.transpose(1, 2) # (batch_size, num_heads, seq_len, hidden_size // num_heads)
        key = key.transpose(1, 2) # (batch_size, num_heads, seq_len, hidden_size // num_heads)
        value = value.transpose(1, 2) # (batch_size, num_heads, seq_len, hidden_size // num_heads)

        # attention
        # compute Q * K^T
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.num_heads) # (batch_size, num_heads, seq_len, seq_len)
        attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        # compute softmax(Q * K^T / sqrt(d_k)) * V
        attention = torch.matmul(attention, value)

        # transpose
        attention = attention.transpose(1, 2)

        # concat heads
        attention = attention.contiguous().view(batch_size, -1, self.hidden_size)

        # output projection
        attention = self.output_projection(attention)

        return attention


class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            dropout: float,
    ):
        super().__init__()

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feed forward
        x = self.feed_forward(x)

        # dropout
        x = self.dropout(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout: float,
    ):
        super().__init__()

        # self attention
        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)

        # feed forward
        self.feed_forward = FeedForward(hidden_size, dropout)

        # layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # self attention
        x = self.self_attention(x, x, x, mask)

        # layer normalization
        x = self.layer_norm(x)

        # feed forward
        x = self.feed_forward(x)

        # layer normalization
        x = self.layer_norm(x)

        return x
