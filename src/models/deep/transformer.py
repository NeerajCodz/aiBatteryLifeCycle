"""
src.models.deep.transformer
============================
Transformer-based models for battery lifecycle prediction (PyTorch).

Architectures:
1. BatteryGPT — Nano Transformer (from reference: 2 encoder layers, 4 heads)
2. Temporal Fusion Transformer (TFT) — Variable selection + GRN + MHA
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# 1. BatteryGPT — Nano Transformer for capacity-sequence prediction
# ═════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BatteryGPT(nn.Module):
    """Nano Transformer for battery capacity sequence prediction.

    Architecture (from reference notebook):
    - Input projection: Linear(input_dim → d_model) * √d_model
    - Sinusoidal positional encoding
    - TransformerEncoder: n_layers encoder layers, n_heads attention heads
    - Output: Linear(d_model → 1) on last time-step
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        (B,) — scalar prediction (next-step capacity or SOH)
        """
        x = self.input_proj(x) * self.scale  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                   # (B, T, d_model)
        out = self.decoder(x[:, -1, :])        # (B, 1) — last time-step
        return out.squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Temporal Fusion Transformer (TFT)
# ═════════════════════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) — core building block of TFT."""

    def __init__(self, d_model: int, d_hidden: int | None = None,
                 d_context: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.context_proj = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        if self.context_proj is not None and context is not None:
            x = x + self.context_proj(context)
        x = self.elu(x)
        x = self.dropout(self.fc2(x))
        gate = torch.sigmoid(self.gate(x))
        x = gate * x
        return self.layer_norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network — learned feature importance weights."""

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.grn_per_var = nn.ModuleList([
            GatedResidualNetwork(d_model, dropout=dropout) for _ in range(n_features)
        ])
        self.grn_softmax = GatedResidualNetwork(n_features * d_model, d_hidden=d_model, dropout=dropout)
        self.softmax_proj = nn.Linear(n_features * d_model, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, n_features, d_model) or (B, n_features, d_model)

        Returns
        -------
        selected : same leading dims + (d_model,)
        weights : (..., n_features)
        """
        orig_shape = x.shape
        # Process each variable through its own GRN
        var_outputs = []
        for i in range(self.n_features):
            var_outputs.append(self.grn_per_var[i](x[..., i, :]))
        var_outputs = torch.stack(var_outputs, dim=-2)  # (..., n_features, d_model)

        # Variable selection weights
        flat = x.reshape(*orig_shape[:-2], -1)  # (..., n_features * d_model)
        weights = F.softmax(self.softmax_proj(flat), dim=-1)  # (..., n_features)

        # Weighted sum
        selected = (var_outputs * weights.unsqueeze(-1)).sum(dim=-2)  # (..., d_model)
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer for battery lifecycle prediction.

    Architecture:
    - Per-feature embedding (Linear per feature → d_model)
    - Variable Selection Network for feature importance
    - LSTM encoder for local temporal processing
    - Multi-Head Self-Attention for long-range dependencies
    - GRN-based output layer

    Input: (B, T, F) — T timesteps, F features
    Output: (B,) — scalar SOH/RUL prediction
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature linear embedding
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

        # Variable selection
        self.var_selection = VariableSelectionNetwork(n_features, d_model, dropout)

        # Local LSTM processing
        self.lstm = nn.LSTM(d_model, d_model, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.lstm_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.lstm_norm = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.mha_norm = nn.LayerNorm(d_model)

        # Output
        self.grn_out = GatedResidualNetwork(d_model, dropout=dropout)
        self.output_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape

        # Embed each feature separately
        embedded = []
        for i in range(F):
            embedded.append(self.feature_embeddings[i](x[:, :, i:i+1]))
        embedded = torch.stack(embedded, dim=-2)  # (B, T, F, d_model)

        # Variable selection
        selected, self.var_weights = self.var_selection(embedded)  # (B, T, d_model)

        # LSTM encoder
        lstm_out, _ = self.lstm(selected)
        gated = self.lstm_gate(lstm_out) * lstm_out
        temporal = self.lstm_norm(selected + self.dropout(gated))

        # Multi-head attention
        attn_out, self.attn_weights = self.mha(temporal, temporal, temporal)
        gated_attn = self.mha_gate(attn_out) * attn_out
        enriched = self.mha_norm(temporal + self.dropout(gated_attn))

        # Output (use last time step)
        out = self.grn_out(enriched[:, -1, :])
        return self.output_head(out).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Attention visualization helper
# ═════════════════════════════════════════════════════════════════════════════

def extract_attention_weights(model: BatteryGPT | TemporalFusionTransformer) -> dict:
    """Extract attention weights for visualization after a forward pass."""
    weights = {}
    if isinstance(model, TemporalFusionTransformer):
        if hasattr(model, "var_weights"):
            weights["variable_selection"] = model.var_weights.detach().cpu().numpy()
        if hasattr(model, "attn_weights"):
            weights["self_attention"] = model.attn_weights.detach().cpu().numpy()
    return weights
