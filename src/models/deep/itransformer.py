"""
src.models.deep.itransformer
=============================
iTransformer family models for SOH prediction (TensorFlow/Keras).

Architectures:
1. iTransformer — Feature-wise MHA → Token-wise MHA → Conv1D FF
2. Physics-Informed iTransformer — Dual-head with physics branch
3. Dynamic-Graph iTransformer — GNN fusion with dynamic adjacency

All models:
- Input: (batch, seq_len, n_features)
- Output: (batch, 1) — SOH prediction
"""

from __future__ import annotations

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ═════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═════════════════════════════════════════════════════════════════════════════

class FeatureWiseMHA(layers.Layer):
    """Feature-wise (inverted) Multi-Head Attention.

    Transposes so that features attend to each other across time.
    Input: (B, T, F) → transpose to (B, F, T) → MHA over F dim → transpose back.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout,
        )
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        # x: (B, T, F) → transpose to (B, F, T) for feature-wise attention
        x_t = tf.transpose(x, perm=[0, 2, 1])  # (B, F, T)
        attn = self.mha(x_t, x_t, training=training)
        attn = self.dropout(attn, training=training)
        out = self.norm(x_t + attn)
        return tf.transpose(out, perm=[0, 2, 1])  # back to (B, T, F)


class TokenWiseMHA(layers.Layer):
    """Token-wise (standard) Multi-Head Attention along time axis."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout,
        )
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        attn = self.dropout(attn, training=training)
        return self.norm(x + attn)


class Conv1DFeedForward(layers.Layer):
    """Conv1D feed-forward network with residual connection."""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        d_ff = d_ff or d_model * 4
        self.conv1 = layers.Conv1D(d_ff, kernel_size=1, activation="gelu")
        self.conv2 = layers.Conv1D(d_model, kernel_size=1)
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        ff = self.conv1(x)
        ff = self.dropout(ff, training=training)
        ff = self.conv2(ff)
        ff = self.dropout(ff, training=training)
        return self.norm(x + ff)


# ═════════════════════════════════════════════════════════════════════════════
# 1. iTransformer
# ═════════════════════════════════════════════════════════════════════════════

def build_itransformer(
    seq_len: int,
    n_features: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_blocks: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
) -> keras.Model:
    """Build iTransformer model for SOH prediction.

    Architecture: input → [FeatureWise-MHA → TokenWise-MHA → Conv1D-FF] × N → GAP → Dense → 1
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="input_seq")

    # Project features to d_model
    x = layers.Dense(d_model, name="input_proj")(inputs)

    for i in range(n_blocks):
        x = FeatureWiseMHA(d_model, n_heads, dropout, name=f"feat_mha_{i}")(x)
        x = TokenWiseMHA(d_model, n_heads, dropout, name=f"token_mha_{i}")(x)
        x = Conv1DFeedForward(d_model, d_ff, dropout, name=f"conv_ff_{i}")(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout, name="fc_drop")(x)
    output = layers.Dense(1, name="soh_output")(x)

    model = keras.Model(inputs, output, name="iTransformer")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 2. Physics-Informed iTransformer
# ═════════════════════════════════════════════════════════════════════════════

def build_physics_itransformer(
    seq_len: int,
    n_features: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_blocks: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    lambda_phy: float = 0.3,
) -> keras.Model:
    """Physics-Informed iTransformer with dual output heads.

    ML Head: iTransformer blocks → Dense → SOH_ml
    Physics Head: |cumulative_current| → MLP → SOH_phy

    Training loss: L = L_ml + λ_phy × L_phy
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="input_seq")

    # ── ML Branch (iTransformer) ─────────────────
    x = layers.Dense(d_model, name="ml_proj")(inputs)
    for i in range(n_blocks):
        x = FeatureWiseMHA(d_model, n_heads, dropout, name=f"ml_feat_{i}")(x)
        x = TokenWiseMHA(d_model, n_heads, dropout, name=f"ml_token_{i}")(x)
        x = Conv1DFeedForward(d_model, d_ff, dropout, name=f"ml_ff_{i}")(x)
    x = layers.GlobalAveragePooling1D(name="ml_gap")(x)
    x = layers.Dense(128, activation="relu", name="ml_fc")(x)
    x = layers.Dropout(dropout)(x)
    soh_ml = layers.Dense(1, name="soh_ml")(x)

    # ── Physics Branch ───────────────────────────
    # Extract current feature (index 1) → abs cumulative sum
    current = AbsCumCurrentLayer(name="abs_cum_current")(inputs)
    p = layers.GlobalAveragePooling1D(name="phy_gap")(current)
    p = layers.Dense(64, activation="relu", name="phy_fc1")(p)
    p = layers.Dense(32, activation="relu", name="phy_fc2")(p)
    soh_phy = layers.Dense(1, name="soh_phy")(p)

    model = keras.Model(inputs, [soh_ml, soh_phy], name="PhysicsInformed_iTransformer")
    return model


class AbsCumCurrentLayer(layers.Layer):
    """Extracts current feature (index 1) and computes abs cumulative sum."""

    def call(self, x, training=False):
        return tf.abs(tf.cumsum(x[:, :, 1:2], axis=1))

    def get_config(self):
        return super().get_config()


class PhysicsInformedLoss(keras.losses.Loss):
    """Combined ML + Physics loss with λ weighting."""

    def __init__(self, lambda_phy: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.lambda_phy = lambda_phy
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred_list):
        soh_ml, soh_phy = y_pred_list
        loss_ml = self.mae(y_true, soh_ml)
        loss_phy = self.mae(y_true, soh_phy)
        return loss_ml + self.lambda_phy * loss_phy


# ═════════════════════════════════════════════════════════════════════════════
# 3. Dynamic-Graph iTransformer
# ═════════════════════════════════════════════════════════════════════════════

class DynamicGraphConv(layers.Layer):
    """Dynamic graph convolution with correlation-based adjacency."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Dense(d_model)
        self.norm = layers.LayerNormalization()

    def call(self, x, training=False):
        """
        x: (B, T, F) — compute feature correlation matrix (F, F) as adjacency
        """
        # Feature-level correlation as adjacency
        # x_t: (B, F, T) → compute (B, F, F) correlation
        x_t = tf.transpose(x, perm=[0, 2, 1])
        x_norm = x_t - tf.reduce_mean(x_t, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x_t, axis=-1, keepdims=True) + 1e-8
        x_norm = x_norm / std
        adj = tf.matmul(x_norm, x_norm, transpose_b=True) / tf.cast(tf.shape(x)[-2], tf.float32)
        adj = tf.nn.softmax(adj, axis=-1)  # (B, F, F)

        # Graph convolution: aggregate features
        x_agg = tf.matmul(adj, x_t)  # (B, F, T)
        x_agg = tf.transpose(x_agg, perm=[0, 2, 1])  # (B, T, F)

        out = self.proj(x_agg)
        return self.norm(x + out)


def build_dynamic_graph_itransformer(
    seq_len: int,
    n_features: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_blocks: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
) -> keras.Model:
    """Dynamic-Graph iTransformer with GNN-Transformer fusion.

    Architecture: input → DynGraphConv → [FeatureWise-MHA → TokenWise-MHA → Conv1D-FF] × N → GAP → Dense → 1
    """
    inputs = keras.Input(shape=(seq_len, n_features))

    # Dynamic graph convolution
    x = DynamicGraphConv(n_features, name="dyn_graph")(inputs)

    # Project to d_model
    x = layers.Dense(d_model, name="proj")(x)

    for i in range(n_blocks):
        x = FeatureWiseMHA(d_model, n_heads, dropout, name=f"dg_feat_{i}")(x)
        x = TokenWiseMHA(d_model, n_heads, dropout, name=f"dg_token_{i}")(x)
        x = Conv1DFeedForward(d_model, d_ff, dropout, name=f"dg_ff_{i}")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1)(x)

    return keras.Model(inputs, output, name="DynamicGraph_iTransformer")
