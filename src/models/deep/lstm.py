"""
src.models.deep.lstm
====================
LSTM / GRU family models for battery lifecycle sequence prediction.

Architectures:
1. Vanilla LSTM — 2-layer, unidirectional
2. Bidirectional LSTM — 2-layer
3. GRU — 2-layer
4. Stacked LSTM with Additive Attention — 3-layer + attention

All models accept input shape ``(batch, seq_len, n_features)`` and
output a single scalar prediction (SOH or RUL).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Vanilla LSTM ─────────────────────────────────────────────────────────
class VanillaLSTM(nn.Module):
    """Standard 2-layer LSTM with final hidden → linear head."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        h_last = self.dropout(h_n[-1])  # (B, H)
        return self.fc(h_last).squeeze(-1)  # (B,)


# ── 2. Bidirectional LSTM ────────────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
    """Bidirectional 2-layer LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        # Concatenate last forward + backward hidden states
        h_fwd = h_n[-2]  # last layer forward
        h_bwd = h_n[-1]  # last layer backward
        h_cat = self.dropout(torch.cat([h_fwd, h_bwd], dim=-1))
        return self.fc(h_cat).squeeze(-1)


# ── 3. GRU ───────────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    """2-layer GRU with linear head."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        h_last = self.dropout(h_n[-1])
        return self.fc(h_last).squeeze(-1)


# ── 4. Stacked LSTM with Additive Attention ─────────────────────────────────
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_outputs : (B, T, H)

        Returns
        -------
        context : (B, H)
        attn_weights : (B, T)
        """
        energy = torch.tanh(self.W(lstm_outputs))  # (B, T, H)
        scores = self.v(energy).squeeze(-1)          # (B, T)
        attn_weights = F.softmax(scores, dim=-1)     # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)  # (B, H)
        return context, attn_weights


class AttentionLSTM(nn.Module):
    """3-layer stacked LSTM with additive attention and linear head."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = AdditiveAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)           # (B, T, H)
        context, self.attn_weights = self.attention(lstm_out)  # (B, H)
        context = self.dropout(context)
        return self.fc(context).squeeze(-1)  # (B,)


# ── MC Dropout inference ────────────────────────────────────────────────────
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make predictions with MC Dropout for uncertainty estimation.

    Parameters
    ----------
    model : nn.Module
        Model with Dropout layers.
    x : torch.Tensor
        Input batch.
    n_samples : int
        Number of stochastic forward passes.

    Returns
    -------
    mean : (B,)
        Mean prediction across samples.
    std : (B,)
        Standard deviation (uncertainty).
    """
    model.train()  # Enable dropout
    preds = torch.stack([model(x) for _ in range(n_samples)])  # (S, B)
    model.eval()
    return preds.mean(dim=0), preds.std(dim=0)


# ── Training utilities ──────────────────────────────────────────────────────
class EarlyStopping:
    """Early stopping with patience and best-model checkpoint."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    max_epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 20,
    device: str | torch.device = "cpu",
    grad_clip: float = 1.0,
) -> dict:
    """Generic training loop for all LSTM/GRU family models.

    Returns dict with train_losses, val_losses, best_epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.L1Loss()  # MAE
    early_stop = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                n_val += 1
        val_losses.append(val_loss / max(n_val, 1))

        scheduler.step()

        if early_stop.step(val_losses[-1], model):
            break

    early_stop.load_best(model)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": len(train_losses) - early_stop.counter,
        "epochs_trained": len(train_losses),
    }
