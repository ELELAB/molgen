import torch
import torch.nn as nn

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

    def forward(self, x, encoder_out, conditions_input, src_mask=None, trg_mask=None):
        x2 = self.layer_norm1(x)
        x2 = self.masked_multi_head_attention(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x2)
        x2 = self.layer_norm2(x)

        if src_mask is not None:
            condition_mask = torch.unsqueeze(conditions_input, -2)
            condition_mask = torch.ones_like(condition_mask, dtype=bool)
            src_mask = torch.cat([condition_mask, src_mask], dim=2)

        x2 = self.multi_head_attention(x2, encoder_out, encoder_out, src_mask)
        x = x + self.dropout_2(x2)
        x_ff = self.layer_norm3(x)
        x_ff = self.feed_forward(x_ff)
        x = x + self.dropout_3(x_ff)
        return x


class GCT(nn.Module):
    def __init__(
        self,
        max_selfie_len,
        n_alphabet_elements,
        n_encoder_blocks,
        n_decoder_blocks,
        d_model,
        d_ff,
        d_latent_space,
        n_mha_heads,
        dropout_p=0.1,
        normalizer_eps=1e-6,
        include_bias=False,
        include_conditions_encoder=False,
        include_conditions_decoder=False,
        include_conditions_reparameterization=False,
    ):
        super().__init__()

        self.smile_embedder = nn.Embedding(n_alphabet_elements, d_model)
        self.attribute_embedder = nn.Linear(1, d_model, bias=False)
        self.positional_encoder = PositionalEncoder(d_model, max_selfie_len + 2, dropout_p)
        self.encoder_blocks = get_clones(
            PreLNEncoder(d_model, n_mha_heads, attention, d_ff, dropout_p, normalizer_eps), n_encoder_blocks
        )
        self.decoder_blocks = get_clones(
            PreLNDecoder(d_model, n_mha_heads, attention, d_ff, dropout_p, normalizer_eps), n_decoder_blocks
        )

        self.layer_norm1 = Normalizer(d_model, normalizer_eps)

        self.linear_mu = nn.Linear(d_model, d_latent_space, bias=include_bias)
        self.linear_logvar = nn.Linear(d_model, d_latent_space, bias=include_bias)

        self.encoder_to_decoder = nn.Linear(d_latent_space, d_model, bias=include_bias)

        self.linear_output = nn.Linear(d_model, n_alphabet_elements, bias=include_bias)

        self.linear_output_attributes = nn.Linear(d_model, 1, bias=include_bias)

        self.d_latent_space = d_latent_space

        self.include_conditions_encoder = include_conditions_encoder
        self.include_conditions_decoder = include_conditions_decoder
        self.include_conditions_reparameterization = include_conditions_reparameterization

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def reparameterize(self, mu, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mu.device, dtype=mu.dtype)
        return z.mul(std) + mu

    def encode(self, smile, conditions=None, mask=None):
        x = self.smile_embedder(smile)
        x = self.positional_encoder(x)

        if self.include_conditions_encoder:
            conditions = self.attribute_embedder(conditions)
            x = torch.cat([conditions, x], dim=1)

        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x, mask)

        x = self.layer_norm1(x)

        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, target_input, conditions_input, z, src_mask, trg_mask):
        x = self.smile_embedder(target_input)
        x = self.positional_encoder(x)
        conditions = self.attribute_embedder(conditions_input)
        if self.include_conditions_decoder:
            x = torch.cat([conditions, x], dim=1)

        encoder_out = self.encoder_to_decoder(z)
        if self.include_conditions_reparameterization:
            encoder_out = torch.cat([conditions, encoder_out], dim=1)

        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, encoder_out, conditions_input, src_mask, trg_mask)

        return self.layer_norm1(x)

    def forward(self, smile, target_input, conditions, src_mask=None, trg_mask=None):
        # Making conditions 3 dimensional
        conditions = conditions.unsqueeze(-1)
        z, mu, logvar = self.encode(smile, conditions, src_mask)
        decoder_out = self.decode(target_input, conditions, z, src_mask, trg_mask)
        if self.include_conditions_decoder:
            decoder_attributes = decoder_out[:, : conditions.shape[1]]
            decoder_selfie = decoder_out[:, conditions.shape[1] :]
            attribute_output = self.linear_output_attributes(decoder_attributes)
        else:
            decoder_selfie = decoder_out
            attribute_output = None
        output = self.linear_output(decoder_selfie)
        return output, attribute_output, z, mu, logvar

    # def generate(self, config, n_samples=16, device="cpu", method = "beam_search", z=None, conditions=None, scale_conditions=True):
    #     if config.additional_metrics.scaling == "robustscaler":
    #         scaler = joblib.load(config.additional_metrics.robust_scaler_path)
    #     else:
    #         assert 1 == 0, "In GCT Generate, implement other scaling methods than robustscaler if this is no longer used."

    #     if type(conditions) == type(None):
    #         conditions = torch.Tensor(get_random_conditions(n_samples, config)).to(device)
    #         scale_conditions = False
    #     else:
    #         conditions = conditions.view(conditions.shape[0], -1)
    #         if conditions.shape[1] == 1:
    #             conditions = conditions.expand(-1, n_samples)
    #             conditions = torch.transpose(conditions,0,1)
    #     if scale_conditions:
    #         conditions = torch.Tensor(scaler.transform(conditions)).to(device)

    #     if z == None:
    #         z_shape = [n_samples, config.data.max_molecule_length + config.additional_metrics.n_metrics, self.d_latent_space]
    #         mu = torch.zeros(z_shape).to(device)
    #         logvar = torch.zeros(z_shape).to(device)
    #         z = self.reparameterize(mu, logvar)

    #     dataset = SmilesGeneratorDataset(z, conditions, config.data.max_molecule_length+1, self.smile_to_int)
    #     dataloader = DataLoader(dataset, batch_size=config.gct.paper_batch_size, shuffle=False)

    #     all_smiles = torch.zeros([n_samples, config.data.max_molecule_length])

    #     for idx, (z_batch, trg_input, conditions_batch) in enumerate(dataloader):
    #         z_batch = z_batch.to(device)
    #         trg_input = trg_input.to(device)
    #         conditions_batch = conditions_batch.to(device)
    #         if method == "beam_search":
    #             smiles = beam_search(self, z_batch, trg_input, conditions_batch, config.gct.beam_search_width, config.gct.beam_search_alpha, config)
    #         elif method == "greedy_search":
    #             smiles = greedy_search(self, z_batch, trg_input, conditions_batch, config)
    #         all_smiles[idx*config.gct.paper_batch_size:(idx+1)*config.gct.paper_batch_size] = smiles
    #     return all_smiles, z, conditions
