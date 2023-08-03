import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.molgen.data.dataset import SelfieGeneratorDataset
from src.molgen.generation.methods import beam_search, greedy_search
from src.molgen.models.transformer import (
    FeedForward,
    MultiHeadAttention,
    Normalizer,
    PositionalEncoder,
    attention,
)
from src.molgen.models.utils import get_clones


class PreLNEncoder(nn.Module):
    def __init__(self, d_model, n_mha_heads, attention_function, d_ff, dropout_p=0.1, normalizer_eps=1e-6):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(n_mha_heads, d_model, attention_function, dropout_p)
        self.layer_norm1 = Normalizer(d_model, normalizer_eps)
        self.layer_norm2 = Normalizer(d_model, normalizer_eps)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        x_mha = self.layer_norm1(x)
        x_mha = self.multi_head_attention(x_mha, x_mha, x_mha, mask)
        x = x + self.dropout1(x_mha)
        x_ff = self.layer_norm2(x)
        x_ff = self.feed_forward(x_ff)
        x = x + self.dropout2(x_ff)
        return x


class PreLNDecoder(nn.Module):
    def __init__(self, d_model, n_mha_heads, attention_function, d_ff, dropout_p=0.1, normalizer_eps=1e-6):
        super().__init__()

        self.masked_multi_head_attention = MultiHeadAttention(n_mha_heads, d_model, attention_function, dropout_p)
        self.multi_head_attention = MultiHeadAttention(n_mha_heads, d_model, attention_function, dropout_p)

        self.layer_norm1 = Normalizer(d_model, normalizer_eps)
        self.layer_norm2 = Normalizer(d_model, normalizer_eps)
        self.layer_norm3 = Normalizer(d_model, normalizer_eps)

        self.dropout_1 = nn.Dropout(dropout_p)
        self.dropout_2 = nn.Dropout(dropout_p)
        self.dropout_3 = nn.Dropout(dropout_p)

        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)

    def forward(self, x, encoder_out, src_mask=None, trg_mask=None):
        x2 = self.layer_norm1(x)
        x2 = self.masked_multi_head_attention(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x2)
        x2 = self.layer_norm2(x)

        x2 = self.multi_head_attention(x2, encoder_out, encoder_out, src_mask)
        x = x + self.dropout_2(x2)
        x_ff = self.layer_norm3(x)
        x_ff = self.feed_forward(x_ff)
        x = x + self.dropout_3(x_ff)
        return x


class ConditionalTransformerDecoder(nn.Module):
    def __init__(
        self,
        max_selfie_len,
        n_alphabet_elements,
        n_decoder_blocks,
        d_model,
        d_ff,
        n_mha_heads_decoder,
        dropout_p=0.1,
        normalizer_eps=1e-6,
        include_bias=False,
        n_attributes=9,
    ):
        super().__init__()

        # Defining the the embedding layers
        self.smile_embedder = nn.Embedding(n_alphabet_elements, d_model)
        self.attribute_embedder = nn.Linear(1, d_model, bias=False)

        # Defining the positional encoder
        self.positional_encoder = PositionalEncoder(d_model, max_selfie_len + 2, dropout_p)

        # Defining Decoder blocks
        self.decoder_blocks = get_clones(
            PreLNDecoder(d_model, n_mha_heads_decoder, attention, d_ff, dropout_p, normalizer_eps), n_decoder_blocks
        )

        # Defining the first layer norm
        self.layer_norm1 = Normalizer(d_model, normalizer_eps)

        # Defining the linear layer mapping the decoder to the output
        self.linear_output = nn.Linear(d_model, n_alphabet_elements, bias=include_bias)

        # Saving the number of attributes
        self.n_attributes = n_attributes

        # Initializing the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def decode(self, target_input, conditions_input, src_mask, trg_mask):
        x = self.smile_embedder(target_input)
        x = self.positional_encoder(x)
        conditions = self.attribute_embedder(conditions_input)

        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, conditions, src_mask, trg_mask)

        return self.layer_norm1(x)

    def forward(self, target_input, conditions, src_mask=None, trg_mask=None):
        # Making conditions 3 dimensional
        conditions = conditions.unsqueeze(-1)
        decoder_out = self.decode(target_input, conditions, src_mask, trg_mask)

        output = self.linear_output(decoder_out)
        return output

    def generate(
        self,
        conditions,
        scaler,
        max_selfie_length,
        symbol_to_index,
        batch_size,
        config,
        n_samples=16,
        device="cpu",
        method="beam_search",
        z=None,
        scale_conditions=True,
    ):
        # If conditions should be scaled before generating (If a scalar was used during training and conditions input here are not scaled)
        NotImplementedError("Not implemented yet")

        if scale_conditions and scaler is not None:
            conditions = torch.Tensor(scaler.transform(conditions)).to(device)

        if z is None:
            z_shape = [n_samples, max_selfie_length, self.d_latent_space]
            mu = torch.zeros(z_shape).to(device)
            logvar = torch.zeros(z_shape).to(device)
            z = self.reparameterize(mu, logvar)

        dataset = SelfieGeneratorDataset(z, conditions, max_selfie_length, symbol_to_index)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_smiles = torch.zeros([n_samples, max_selfie_length]).to(device)

        self.eval()

        for idx, (z_batch, trg_input, conditions_batch) in enumerate(dataloader):
            z_batch = z_batch.to(device)
            trg_input = trg_input.to(device)
            conditions_batch = conditions_batch.to(device)

            with torch.no_grad():
                if method == "beam_search":
                    smiles = beam_search(
                        self,
                        z_batch,
                        trg_input,
                        conditions_batch,
                        config.gct.beam_width,
                        config.gct.beam_search_alpha,
                        symbol_to_index,
                    )
                elif method == "greedy_search":
                    smiles = greedy_search(self, z_batch, trg_input, conditions_batch, symbol_to_index)
            all_smiles[idx * batch_size : (idx + 1) * batch_size] = smiles
        return all_smiles, z, conditions

    def predict_attr_from_z(
        self,
        z,
        conditions,
        scaler,
        max_selfie_length,
        symbol_to_index,
        batch_size,
        config,
        n_samples=16,
        device="cpu",
        method="beam_search",
        scale_conditions=True,
    ):
        NotImplementedError("Not implemented yet")

        # Defining the device
        device = z.device

        # Generating the smiles from the z
        smiles, _, _ = self.generate(
            conditions,
            scaler,
            max_selfie_length,
            symbol_to_index,
            batch_size,
            config,
            n_samples,
            device,
            method,
            z,
            scale_conditions,
        )

        # Getting the batch size
        batch_size = z.shape[0]

        # Convert conditions to float
        conditions = conditions.type(torch.float32)

        # Making the target input from the generated smiles
        trg_input = smiles.type(torch.int64)

        self.train()

        decoder_out = self.decode(trg_input, conditions.unsqueeze(-1), z, src_mask=None, trg_mask=None)
        if self.include_conditions_decoder:
            decoder_attributes = decoder_out[:, : conditions.shape[1]]
            decoder_selfie = decoder_out[:, conditions.shape[1] :]
            self.linear_output_attributes(decoder_attributes)
        else:
            decoder_selfie = decoder_out
        attributes_pred = self.linear_attributes_decoder(decoder_selfie.view(decoder_selfie.shape[0], -1))

        return attributes_pred, smiles

        # x = self.positional_encoder(x)

        # if self.include_conditions_encoder:
        #     conditions = self.attribute_embedder(conditions)
        #     x = torch.cat([conditions, x], dim=1)

        # for i in range(len(self.encoder_blocks)):
        #     x = self.encoder_blocks[i](x, mask)

        # x = self.layer_norm1(x)

        # mu = self.linear_mu(x)
        # logvar = self.linear_logvar(x)
        # attributes = self.linear_attributes(x.view(x.shape[0], -1))

        # z = self.reparameterize(mu, logvar)

        # else:
        #     batch_size = z.shape[0]
        #     conditions = conditions.type(torch.float32)
        #     trg_input = torch.ones([max_selfie_length]) * symbol_to_index["[nop]"]
        #     trg_input = trg_input.type(torch.int64)
        #     trg_input = trg_input.unsqueeze(0).to(device)
        #     trg_input = trg_input.repeat(batch_size, 1)

        #     self.train()
        #     if method == "beam_search":
        #         smiles = beam_search(self, z, trg_input, conditions, config.gct.beam_width, config.gct.beam_search_alpha, symbol_to_index, config)
        #     elif method == "greedy_search":
        #         smiles = greedy_search(self, z, trg_input, conditions, symbol_to_index, config)

        #     return smiles, z, conditions
