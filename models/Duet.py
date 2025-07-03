import torch
from torch import nn
import numpy as np
from einops import rearrange
from layers.Masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
from layers.linear_extractor_cluster import Linear_extractor_cluster
import copy

class Model(nn.Module):
    def __init__(self, config, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = config.task_name
        
        self.cluster = Linear_extractor_cluster(config)
        
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        
        self.pred_len = config.pred_len
                
        self.args = config

        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=0,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation="gelu",
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, self.pred_len), nn.Dropout(config.fc_dropout))

    def forecast(self, input, x_mark_enc, x_dec, x_mark_dec):
        
        means = input.mean(1, keepdim=True).detach()
        input = input - means
        stdev = torch.sqrt(
            torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input /= stdev

        # x: [batch_size, seq_len, n_vars]

        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)
            
        # temporal_feature = input

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        # if self.n_vars > 1:
        #     changed_input = rearrange(input, 'b l n -> b n l')
        #     channel_mask = self.mask_generator(changed_input)

        #     channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

        #     output = self.linear_head(channel_group_feature)
        # else:
        output = temporal_feature
        output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        
        output = output * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, joint=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None