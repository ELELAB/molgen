import copy

import torch
import torch.nn as nn


def make_padding_mask(x, padding_int, n_conditions=0):
    """
    This function makes padding mask to be used in the self attention parts of the transformer.
    It masks all elements equal to the padding integer.
    The n_conditions input makes sure to append n columns to the start of the mask which are allways visible.
    """
    x = x.view(-1, x.shape[-1])
    mask = torch.ones(x.shape[0], x.shape[1])
    mask[x == padding_int] = 0
    if n_conditions > 0:
        condition_mask = torch.ones([x.shape[0], n_conditions])
        mask = torch.cat([condition_mask, mask], axis=1)
    mask = mask.unsqueeze(-2)
    return mask


def make_nopeak_mask(batch_size, dimension=80):
    """
    This function makes the nopeak mask for the decoder part of the transformer.
    """
    mask = torch.tril(torch.ones([dimension, dimension])).expand(batch_size, dimension, dimension)
    return mask


def get_clones(module, N):
    """
    Takes in a module and outputs a modulelist with the module copied N times.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
