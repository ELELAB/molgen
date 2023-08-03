import copy

import torch
import torch.nn as nn


def make_padding_mask(x, padding_idx, n_conditions=0):
    """
    This function makes padding mask to be used in the self attention parts of the transformer.
    It masks all elements equal to the padding integer.
    The n_conditions input makes sure to append n columns to the start of the mask which are allways visible.
    """
    device = x.device

    # Make mask of shape (batch_size, seq_len)
    mask = (x != padding_idx).unsqueeze(-2).to(device)
    mask[:, :, 0] = True

    # Add n_conditions columns to the start of the mask
    if n_conditions > 0:
        condition_mask = torch.ones([x.shape[0], n_conditions], dtype=bool).unsqueeze(-2).to(device)
        mask = torch.cat([condition_mask, mask], axis=2)
    return mask


def make_nopeak_mask(batch_size, device="cpu", dimension=80, n_conditions=0):
    """
    This function makes the nopeak mask for the decoder part of the transformer.
    """

    mask = torch.tril(torch.ones([dimension, dimension])).expand(batch_size, dimension, dimension)

    # Add n_conditions columns to the left and top of the mask
    if n_conditions > 0:
        final_mask = torch.ones([batch_size, dimension + n_conditions, dimension + n_conditions])
        final_mask[:, :, n_conditions:] = 0
        final_mask[:, n_conditions:, n_conditions:] = mask
    else:
        final_mask = mask

    return final_mask.to(device)


def get_clones(module, N):
    """
    Takes in a module and outputs a modulelist with the module copied N times.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
