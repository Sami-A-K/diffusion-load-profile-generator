import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Zerlegt [batch, seq_len, n_series_features] in nicht überlappende Patches der Länge
    patch_len, projiziert jeden Patch linear auf hidden_dim und addiert ein sin/cos-
    Positions-Encoding. Vor dem Patchen wird die letzte Stunde stride-mal angehängt
    (Nie et al., ICLR 2023).

    Args:
        patch_len (int): P, Patch-Länge in Stunden.
        stride (int): S, Schrittweite zwischen Patch-Starts (S=P -> nicht überlappend).
        n_series_features (int): Merkmale pro Stunde vor dem Patchen.
        hidden_dim (int): Zieldimension D nach der Projektion.
    """

    def __init__(self, patch_len, stride, n_series_features, hidden_dim):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(patch_len * n_series_features, hidden_dim)

    def relative_position_encoding(self, seq_len, real_counts, device):
        """
        sin/cos-Positions-Encoding nach dem Abstand zum letzten realen Patch je Sample
        (0 = jüngster Patch). Für Padding-Patches wird der Abstand auf 0 gesetzt; diese sind
        in der Attention maskiert.
        """
        idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        distance = real_counts.to(torch.float32).unsqueeze(1) - 1 - idx.unsqueeze(0)
        distance = distance.clamp(min=0)

        half = self.hidden_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)

        args = distance.unsqueeze(-1) * freqs[None, None, :]
        pos_enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.hidden_dim % 2 == 1:
            pos_enc = torch.cat([pos_enc, torch.zeros_like(pos_enc[..., :1])], dim=-1)

        return pos_enc

    def forward(self, series, padding_mask=None):
        """
        Args:
            series (torch.FloatTensor): [batch, seq_len, n_series_features].
            padding_mask (torch.BoolTensor, optional): [batch, seq_len], True an Padding-Positionen.
        Returns:
            tokens (torch.FloatTensor): [batch, num_patches, hidden_dim].
            patch_padding_mask (torch.BoolTensor or None): [batch, num_patches], True für Patches,
                die nur aus Padding-Stunden bestehen.
        """
        last_hour = series[:, -1:, :].expand(-1, self.stride, -1)
        series = torch.cat([series, last_hour], dim=1)

        patches = series.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(2)

        tokens = self.proj(patches)

        patch_padding_mask = None
        if padding_mask is not None:
            # Die angehängten Stunden übernehmen den Padding-Status der letzten Stunde.
            tail_mask = padding_mask[:, -1:].expand(-1, self.stride)
            padding_mask = torch.cat([padding_mask, tail_mask], dim=1)
            patch_padding_mask = padding_mask.unfold(1, self.patch_len, self.stride).all(dim=-1)
            real_counts = (~patch_padding_mask).sum(dim=1)
        else:
            real_counts = torch.full((tokens.shape[0],), tokens.shape[1], device=series.device)

        tokens = tokens + self.relative_position_encoding(tokens.shape[1], real_counts, series.device)

        return tokens, patch_padding_mask


class PatchTSTEncoderLayer(nn.Module):
    """
    Ein Encoder-Layer nach PatchTST (Nie et al., 2023)

    Args:
        hidden_dim (int): Modelldimension D.
        nhead (int): Anzahl Attention-Heads H.
        dim_feedforward (int): FFN-Zwischendimension F.
        dropout (float): Dropout nach Attention und FFN, jeweils vor dem Residual-Add.

    LayerNorm wird auf den Sublayer-Output vor dem Residual-Add angewendet.
    """

    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        dropout = self.dropout1(attn_out)
        normed = self.norm1(dropout)
        x = x + normed

        ffn = self.ffn(x)
        dropout = self.dropout2(ffn)
        normed = self.norm2(dropout)
        x = x + normed

        return x

class MeasurementSeriesPatchEncoder(nn.Module):
    """
    Args:
        n_series_features (int): Merkmale pro Stunde. Default=4: Last, Temperatur, sin/cos Stunde-des-Tages.
        n_static_features (int): Statische Metadaten. Default=10: One-hot Category
        hidden_dim (int): Modelldimension, konsistent mit hidden_dim des ConditionedDenoiser.
        nhead (int): Anzahl Attention-Heads.
        dim_feedforward (int): FFN-Dimension im Encoder-Layer (Standard-Ratio in PatchTST: 2x hidden_dim).
        num_layers (int): Anzahl Encoder-Layer.
        patch_len (int): Patch-Länge in Stunden (siehe PatchEmbedding).
        stride (int): Patch-Stride in Stunden; stride=patch_len -> nicht überlappende Patches.
    """

    def __init__(self, n_series_features=4, n_static_features=10, hidden_dim=128, nhead=4, dim_feedforward=256, num_layers=3, dropout=0.2, patch_len=6, stride=6):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_len, stride, n_series_features, hidden_dim)
        self.static_proj = nn.Linear(n_static_features, hidden_dim)

        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.hidden_dim = hidden_dim

    def forward(self, series, static, padding_mask=None):
        """
        Args:
            series (torch.FloatTensor): [batch, seq_len, n_series_features], eine Zeile pro Stunde des Beobachtungsfensters.
            static (torch.FloatTensor): [batch, n_static_features], statische Metadaten.
            padding_mask (torch.BoolTensor, optional): [batch, seq_len], True an Padding-Positionen (kürzere Fenster innerhalb desselben Batches).
        Returns:
            tokens (torch.FloatTensor): [batch, num_patches, hidden_dim], Kontext-Tokens für die Cross-Attention.
            patch_padding_mask (torch.BoolTensor or None): [batch, num_patches], True an Padding-Patches;
                wird an den CrossAttentionBlock weitergegeben.
        """
        tokens, patch_padding_mask = self.patch_embed(series, padding_mask)
        tokens = tokens + self.static_proj(static).unsqueeze(1)

        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=patch_padding_mask)

        return tokens, patch_padding_mask
