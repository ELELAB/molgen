import copy

import torch
import torch.nn as nn


def make_padding_mask(x, padding_idx, n_conditions=0):
    """
    This function makes padding mask to be used in the self attention parts of the transformer.
    It masks all elements equal to the padding integer.
    The n_conditions input makes sure to append n columns to the start of the mask which are allways visible.
    """
    # Make mask of shape (batch_size, seq_len)
    mask = torch.ones_like(x, dtype=bool)

    # Find first occurence of padding_int in x
    padding_mask = torch.eq(x, padding_idx)
    first_padding_idx = mask.shape[1] - torch.sum(padding_mask, axis=1) + 1

    # Set all elements of mask after first padding int for each selfie to 0
    for i in range(x.shape[0]):
        mask[i, first_padding_idx[i] + 1 :] = 0

    # Add n_conditions columns to the start of the mask
    if n_conditions > 0:
        condition_mask = torch.ones([x.shape[0], n_conditions])
        mask = torch.cat([condition_mask, mask], axis=1)
    mask = mask.unsqueeze(-2)
    return mask


def make_nopeak_mask(batch_size, dimension=80, n_conditions=0):
    """
    This function makes the nopeak mask for the decoder part of the transformer.
    """
    mask = torch.tril(torch.ones([dimension, dimension])).expand(batch_size, dimension, dimension)

    # Add n_conditions columns to the left and top of the mask
    if n_conditions > 0:
        final_mask = torch.ones([batch_size, dimension + n_conditions, dimension + n_conditions])
        final_mask[:, n_conditions:, n_conditions:] = mask
    else:
        final_mask = mask
    return final_mask


def get_clones(module, N):
    """
    Takes in a module and outputs a modulelist with the module copied N times.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
