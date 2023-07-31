import copy

import torch
import torch.nn.functional as F

from src.molgen.models.utils import make_nopeak_mask, make_padding_mask


def greedy_search(model, z, trg_input, conditions, symbol_to_index):
    """
    DESCRIPTION:
        This function performs the greedy search when generating molecules.
        This means taking the element with highest probabily every time, and not considering any other.

    INPUT:
        model: Model: A model which needs to have a model.decode and model.linear_out function.
                      decode should decode z, trg_input, conditions while linear_out takes the decoded part
                      and gives logits out.

        z: Tensor: The z is the output of the encoder part of the transformer.

        trg_input: Tensor: Should be the beginning of the tensors generated with beam_search. Should only contain
                           <sos> and <pad> tokens.

        conditions: Tensor: Should contain the scaled conditions wanted with the algorithm.

    OUTPUT:
        generated_smiles: TENSOR: A Tensor containing the generated smiles molecules.
    """
    n_elements = z.shape[0]
    device = z.device
    generated_smiles = copy.deepcopy(trg_input)
    eos_not_reached = torch.ones(n_elements, dtype=bool)

    # Making masks
    # Making the masks for the decoder.
    # Defining the masks
    src_mask = None
    if model.include_conditions_decoder:
        trg_no_peak_mask = make_nopeak_mask(
            1, device=device, dimension=trg_input.shape[1], n_conditions=conditions.shape[-1]
        )
        trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=conditions.shape[-1])
    else:
        trg_no_peak_mask = make_nopeak_mask(1, device=device, dimension=trg_input.shape[1], n_conditions=0)
        trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=0)

    trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

    # Iterating and generating a new element untill an eos token has been generated for each of the smiles in the batch
    for i in range(1, trg_input.shape[1]):
        # Getting decoder output, logits, and finding next element
        decoder_out = model.decode(generated_smiles, conditions.unsqueeze(-1), z, src_mask=src_mask, trg_mask=trg_mask)
        decoder_selfie = decoder_out[:, conditions.shape[1] :] if model.include_conditions_decoder else decoder_out
        output = model.linear_output(decoder_selfie)
        logits = output[:, i - 1, :]
        elements = torch.argmax(logits, dim=1)

        # Storing new elements
        generated_smiles[eos_not_reached, i] = elements[eos_not_reached]

        # Checking if eos is reached and terminating if eos is reached in all smiles.
        eos_not_reached = torch.sum(generated_smiles[:, : i + 1] == symbol_to_index["[nop]"], axis=1) < 2
        if torch.sum(eos_not_reached) == 0:
            break

    return generated_smiles


def beam_search(model, z, trg_input, conditions, width, alpha, symbol_to_index):
    """
    DESCRIPTION:
        This function peforms the refined beam search algorithm to generated smile molecules.
        In each iteration it keeps the "width" best elements, computes refined log probabilites for all of them,
        and keeps the "width" best. This is done untill all elements have reached the <eos> token or the length
        exceeds the max length. The best is then returned.
        Refined beam search normalizes the logp by 1/smile_length ** alpha

    INPUT:
        model: Model: A model which needs to have a model.decode and model.linear_out function.
                      decode should decode z, trg_input, conditions while linear_out takes the decoded part
                      and gives logits out.

        z: Tensor: The z is the output of the encoder part of the transformer.

        trg_input: Tensor: Should be the beginning of the tensors generated with beam_search. Should only containing
                           <sos> and <pad> tokens.

        conditions: Tensor: Should contain the scaled conditions wanted with the algorithm.

        width: Int: The Beam Search width, this determines how many elements are taken into consideration in each iteration.

        alpha: Float: Length normalization, if alpha=1 then logp will be normalized by length, if alpha=0 no normalization.

    OUTPUT:
        generated_smiles: TENSOR: A Tensor containing the generated smiles molecules.
    """

    # Width cant be bigger than vocab_size
    n_elements = z.shape[0]
    device = z.device
    generated_smiles = torch.zeros([trg_input.shape[0], trg_input.shape[1]])

    for element in range(n_elements):
        # Extracting the current values of z, input and conditions
        current_z = z[element].view(-1, z.shape[1], z.shape[2])
        current_trg_input = trg_input[element].view(-1, trg_input.shape[-1])
        current_condition = conditions[element].view(-1, conditions.shape[-1])

        # Making the masks for the decoder.
        # Defining the masks
        src_mask = None
        if model.include_conditions_decoder:  # TRG MASK BLIVER FORKERT SIZE bliver 16 x istedet for 1 x
            trg_no_peak_mask = make_nopeak_mask(
                1, device=device, dimension=current_trg_input.shape[1], n_conditions=conditions.shape[-1]
            )
            trg_padding_mask = make_padding_mask(
                current_trg_input, symbol_to_index["[nop]"], n_conditions=conditions.shape[-1]
            )
        else:
            trg_no_peak_mask = make_nopeak_mask(1, device=device, dimension=current_trg_input.shape[1], n_conditions=0)
            trg_padding_mask = make_padding_mask(current_trg_input, symbol_to_index["[nop]"], n_conditions=0)

        trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

        # Getting the logits of the first element
        decoder_out = model.decode(
            current_trg_input, current_condition.unsqueeze(-1), current_z, src_mask=src_mask, trg_mask=trg_mask
        )
        logits_first_element = model.linear_output(decoder_out)[0, 0, :]

        # Getting the top k elements and converting the probabilities to log probabilities
        output_probabilities = F.softmax(logits_first_element, dim=-1)
        best_logits, best_elements = torch.topk(output_probabilities, k=width)
        best_log_probabilities = torch.log(best_logits).unsqueeze(-1)

        # Expanding the trg_input vector to get the trg_input vector "width" times and inputting first found elements.
        current_trg_input = current_trg_input.expand(width, -1) + 0
        current_trg_input[:, 1] = best_elements

        # Expanding z and conditions vector to have them "width" times, this is necessary to run all trg_input through decoder at the same time.
        current_z = current_z.expand(width, -1, -1) + 0
        current_condition = current_condition.expand(width, -1) + 0

        # Making vectors to keep track of which elements have reached the end and theirs lengths until the <eos> token.
        eos_not_reached = torch.ones(width, dtype=bool).to(device)
        smile_lengths = torch.ones(width).to(device)

        for i in range(2, trg_input.shape[1]):
            # Making masks for the decoder.
            src_mask = None
            if model.include_conditions_decoder:
                trg_no_peak_mask = make_nopeak_mask(
                    width, device=device, dimension=current_trg_input.shape[1], n_conditions=conditions.shape[-1]
                )
                trg_padding_mask = make_padding_mask(
                    current_trg_input, symbol_to_index["[nop]"], n_conditions=conditions.shape[-1]
                )
            else:
                trg_no_peak_mask = make_nopeak_mask(
                    width, device=device, dimension=current_trg_input.shape[1], n_conditions=0
                )
                trg_padding_mask = make_padding_mask(current_trg_input, symbol_to_index["[nop]"], n_conditions=0)

            trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

            # Getting the logits of the next output and converting to log probility and adding to the current logp for the smile.
            decoder_out = model.decode(
                current_trg_input, current_condition.unsqueeze(-1), current_z, src_mask=src_mask, trg_mask=trg_mask
            )
            if model.include_conditions_decoder:
                decoder_selfie = decoder_out[:, current_condition.shape[1] :]
            else:
                decoder_selfie = decoder_out
            output = model.linear_output(decoder_selfie)
            logits = output[:, i - 1, :]
            output_probabilities = F.softmax(logits, dim=-1)
            # Only adding logp for the elements not yet reached <eos> therefor multipling with smile_not_reached_end
            log_probabilities = torch.log(output_probabilities) * eos_not_reached.view(-1, 1) + best_log_probabilities

            smile_lengths[eos_not_reached] = i

            # Normalizing probabilities with alpha and smile_lengths
            alpha_log_probabilities = torch.mul(log_probabilities, (1 / smile_lengths**alpha).view(width, -1))

            # Setting all but one value of the ones that has reached eos to min logp so that more than one of these will not be chosen.
            # All logits for this are the same because output_probabilities is multiplied by 0 and therefor log_probabilities = best_log_probabilities
            alpha_log_probabilities[torch.logical_not(eos_not_reached), 1:] = torch.min(alpha_log_probabilities)

            # Getting best elements, flatten needed.
            best_alpha_log_probabilities, best_elements = torch.topk(alpha_log_probabilities.flatten(), k=width)

            # Converting indexes to 2 dimensional
            n_selfie_components = len(symbol_to_index)
            best_cols = best_elements % n_selfie_components
            best_rows = torch.div(best_elements, n_selfie_components, rounding_mode="floor")

            # Updates current_trg_input to contain the smiles from previous steps which are the best after adding element from this iteration.
            current_trg_input = current_trg_input[best_rows]

            # Updating eos_not_reached as those that has reached end might have been overwritten.
            eos_not_reached = eos_not_reached[best_rows]
            smile_lengths = smile_lengths[best_rows]

            # Adding new element to the smiles.
            current_trg_input[eos_not_reached, i] = best_cols[eos_not_reached]

            # Updating best log probilities vector
            best_log_probabilities = log_probabilities[best_rows, best_cols].view(-1, 1)

            # Checking if smiles reached end again if one of the added elements was a <eos> token
            eos_not_reached = torch.sum(current_trg_input[:, : i + 1] == symbol_to_index["[nop]"], axis=1) <= 1
            if torch.sum(eos_not_reached) == 0:
                break

        # Adding the most probable smile to the generated smiles vector
        current_smile = current_trg_input[best_log_probabilities.argmax(), :]
        generated_smiles[element] = current_smile

    return generated_smiles
