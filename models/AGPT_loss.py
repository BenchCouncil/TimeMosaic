from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B*C, num_patch, d_model]
        return self.pe[:, :x.size(1)]

class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len_list, mode='fixed', dropout=0.0, seq_len=96, in_channels=1):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.mode = mode
        self.max_patch_len = max(patch_len_list)
        self.min_patch_len = min(patch_len_list)
        self.region_num = seq_len // self.max_patch_len
        self.d_model = d_model
        self.in_channels = in_channels
        
        self.register_buffer('target_ratio', torch.ones(len(patch_len_list)) / len(patch_len_list))

        self.region_cls = nn.Sequential(
            nn.Linear(self.max_patch_len, 64),
            nn.ReLU(),
            nn.Linear(64, len(patch_len_list))
        )

        self.embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model, bias=False) for patch_len in patch_len_list
        ])

        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        assert L == self.region_num * self.max_patch_len, \
            f"Expected seq_len={self.region_num * self.max_patch_len}, but got {L}"

        x = x.reshape(B*C, self.region_num, self.max_patch_len)  # [B*C, R, max_patch_len]

        all_patches = []
        cls_pred_list = []
        for region_idx in range(self.region_num):
            region = x[:, region_idx, :]  # [B*C, max_patch_len]
            
            # 分类：确定patch长度
            cls_logits = self.region_cls(region)  # [B*C, num_classes]
            cls_pred = torch.argmax(cls_logits, dim=-1)  # [B*C]
            cls_pred_list.append(cls_pred)
            region_patches = []
            
            for idx, patch_len in enumerate(self.patch_len_list):
                selected_idx = (cls_pred == idx).nonzero(as_tuple=True)[0]
                if selected_idx.numel() == 0:
                    continue
                selected_region = region[selected_idx]  # [N, max_patch_len]
                patches = selected_region.unfold(-1, patch_len, patch_len)  # [N, num_patch, patch_len]

                if self.mode == 'fixed':
                    target_patch_num = self.max_patch_len // self.min_patch_len
                    repeat = target_patch_num - patches.size(1)
                    if repeat > 0:
                        patches = patches.repeat_interleave(repeat+1, dim=1)[:, :target_patch_num, :]
                        # patches = F.pad(patches, (0, 0, 0, repeat), mode='constant', value=0.0)
                patches_emb = self.embeddings[idx](patches)  # [N, num_patch, d_model]

                # 放回对应位置
                tmp = torch.zeros(selected_idx.size(0), patches_emb.size(1), self.d_model, device=x.device)
                tmp = patches_emb
                region_patches.append((selected_idx, tmp))

            # 合并所有选中
            region_patches_sorted = torch.zeros(B*C, patches_emb.size(1), self.d_model, device=x.device)
            for idx_group, emb_group in region_patches:
                region_patches_sorted[idx_group] = emb_group

            all_patches.append(region_patches_sorted)

        # 拼接所有区域patch
        x_patch = torch.cat(all_patches, dim=1)  # [B*C, total_num_patch, d_model]
        x_patch += self.position_embedding(x_patch)
        x_patch = self.dropout(x_patch)

        if self.training:
            all_cls_pred = torch.cat(cls_pred_list, dim=0)  # List of [B*C] → [B*C*R]
            counts = torch.bincount(all_cls_pred, minlength=len(self.patch_len_list)).float()
            current_ratio = counts / counts.sum()
            budget_loss = F.mse_loss(current_ratio[:-1], self.target_ratio[:-1])  # Budget Loss
        else:
            budget_loss = None

        return x_patch, C, budget_loss


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.cell_size = 16
        self.num_cells = self.pred_len // self.cell_size

        self.patch_len_list = eval(configs.patch_len_list)

        self.patch_embedding = AdaptivePatchEmbedding(
            d_model=configs.d_model,
            patch_len_list=self.patch_len_list,
            mode='fixed',
            dropout=configs.dropout,
            seq_len=configs.seq_len,
            in_channels=configs.enc_in,
        )

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2)))

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars, budget_loss = self.patch_embedding(x_enc)
        # print('enc_out shape:', enc_out.shape)
        # raise ValueError('enc_out shape:', enc_out.shape)
        # enc_out = self.enc_embedding(x_enc)
        # print('enc_out shape:', enc_out.shape)
        # raise ValueError('enc_out shape:', enc_out.shape)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, budget_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'AGPTp_loss', 'AGPT_loss']:
            dec_out, budget_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.task_name == 'long_term_forecast':
                return dec_out[:, -self.pred_len:, :]
            else:
                return dec_out[:, -self.pred_len:, :], budget_loss
        return None