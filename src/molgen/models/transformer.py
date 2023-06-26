import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def attention(query, keys, values, head_dim, mask=None, dropout=None):
    """
    EXPLANATION:
        Attention function. This function computes the attention between matricies values V, keys K and query Q.
        The formular is: Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_K)) . V, d_K = dimension of the inputs embedding. If more than one head is used this is embed_dim/n_heads

    INPUT:
        query: TENSOR - a Tensor with the values of the query matrix. Dimension is [B x (n_attr + max_l) x embed_dim]
                        B = Batch size, n_attr = number of attributes, max_l = maximum length of smile molecule, embed_dim = Embedding dimension.
        keys: TENSOR - a Tensor with the values of the keys matrix. Dimension is [B x (n_attr + max_l) x embed_dim]
                        B = Batch size, n_attr = number of attributes, max_l = maximum length of smile molecule, embed_dim = Embedding dimension.
        values: TENSOR - a Tensor with the values of the values matrix. Dimension is [B x (n_attr + max_l) x embed_dim]
                        B = Batch size, n_attr = number of attributes, max_l = maximum length of smile molecule, embed_dim = Embedding dimension.

    OUTPUT:
        output: TENSOR - The result of the attention function.

    """
    ############ Calculating dot(Q, K^T)/sqrt(d_K) ###############
    scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
    ############ Applying no peak mask or padding mask ###############
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    ############ Applying softmax - makes masked values equal 0 ###############
    scores = F.softmax(scores, dim=-1)

    ############ Applies dropout ###############
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, values)
    return output


class PositionalEncoder(nn.Module):
    """
    EXPLANATION:
        Positional embedding to make the model understand difference of position in the SMILE string.
        Initializing the class precalculates the matrix. The forward method adds the positional embedding to the input matrix.

    INITIALIZING:
        d_model: INT - The embedding dimension of each element of the SMILE molecules.
        max_seq_len: INT - The maximum number of elements of a SMILE.

    """

    def __init__(self, d_model, max_seq_len=200, dropout_p=0.1):
        """
        DESCRIPTION:
            Creating the positional embedding matrix. Value of the matrix is:
            PE(pos, i) = sin(pos / 10000^(2*i/d_m) if i % 2 == 0
            PE(pos, i) = cos(pos / 10000^(2*i/d_m) if i % 2 == 1

        INPUT:
            d_model: INT - The embedding dimension of each element of the SMILE molecules.
            max_seq_len: INT - The maximum number of elements of a SMILE.

        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout_p)

        ############# Creating the positional embedding matrix #####################
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                if d_model % 2 == 0:
                    pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                    pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
                else:
                    pe[pos, i] = np.sin(pos / (10000 ** ((2 * (i - 1)) / d_model)))
                    if i != d_model - 1:
                        pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        device = x.device
        ############## Makes original matrix larger, to make sure that positional embedding does not have to large of an impact ###########
        x = x * np.sqrt(self.d_model)
        ############## Adding the positional embedding to embedding of the smiles which are the last max_seq_len elements of the matrix. ##########
        batch_size = x.size(0)
        pe = self.pe.repeat(batch_size, 1, 1)
        x_seq_length = x.size(1)
        x = x + pe[:, :x_seq_length].to(device)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    EXPLANATION:
        Multiheaded attention part of a transformer. Given a attention function it gives the output of a multiheaded attention part.
    """

    def __init__(self, heads, d_model, attention_function, dropout=0.1):
        """
        INITIALIZING:
            heads: INT - Number of heads in the mulitheaded attention. d_model has to be divisable with heads.
            d_model: INT - The embedding dimension of each element of the SMILE molecules.
            attention_function: FUNCTION - A function taking VALUES, KEYS, QUERY as input and calculates the Attention SCORES.
        """
        super().__init__()

        self.d_model = d_model
        self.head_dim = d_model // heads
        self.heads = heads
        self.attention_function = attention_function

        if self.head_dim * self.heads != self.d_model:
            raise Exception(
                f"dimension of embeddings, should be divisible by heads, d_model = {self.d_model}, head_dim ="
                f" {self.head_dim}, heads = {self.heads}"
            )

        self.q_linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        The forward pass of the mulitheaded attention. It divides the query, keys, values vectors into heads,
        then it computes scores with the attention function. It concatenates and returns the result.
        """
        batch_size = values.shape[0]

        # Creating values, querys, keys matricies
        values = self.v_linear(values)
        keys = self.k_linear(keys)
        query = self.q_linear(query)

        # Didiving the matricies into heads
        values = values.view(batch_size, -1, self.heads, self.head_dim)
        keys = keys.view(batch_size, -1, self.heads, self.head_dim)
        query = query.view(batch_size, -1, self.heads, self.head_dim)

        # Transpose to gain dimension batch_size * heads * max_length * head_dim
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        query = query.transpose(1, 2)

        # Calculate attention
        scores = self.attention_function(query, keys, values, self.head_dim, mask, self.dropout)

        # Concatenate heads and regain original dimension
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.linear_out(concat)

        return output


class FeedForward(nn.Module):
    """
    Feed forward part of a transformer encoder/decoder block.
    This expands the input, performs RelU/dropout, performs a shrinkage and returns a vector with the same dimension as the input.
    """

    def __init__(self, d_model, d_ff=64, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.layers(x)
        return x


class Normalizer(nn.Module):
    """
    Normalization part of a Transformer encoder/decoder block.
    This scales the input with optimized parameters alpha and bias.
    It also adds a small amount of noise to make the model more robust.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.d_model = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
