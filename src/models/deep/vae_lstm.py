"""
src.models.deep.vae_lstm
========================
Variational Autoencoder with LSTM encoder/decoder for battery health
state embedding and anomaly detection.

Architecture:
- Encoder: 2-layer bi-LSTM → μ and log-σ (latent dim)
- Reparameterization: z = μ + ε·σ
- Decoder: 2-layer LSTM → reconstruct input sequence
- Health head: latent μ → MLP → SOH/RUL prediction
- Anomaly: cycles with reconstruction error > 3σ flagged

Loss: L = L_recon + β·L_KL   (β annealed from 0→1 during training)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_LSTM(nn.Module):
    """Variational Autoencoder with LSTM backbone for battery sequences."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # ── Encoder ──
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # ── Decoder ──
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

        # ── Health prediction head ──
        self.health_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent distribution parameters."""
        out, (h_n, _) = self.encoder_lstm(x)
        # Concatenate last forward and backward hidden states
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*H)
        mu = self.fc_mu(h_cat)          # (B, latent_dim)
        logvar = self.fc_logvar(h_cat)  # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed sequence."""
        # Repeat latent vector across sequence length
        z_proj = self.decoder_input(z)                    # (B, H)
        z_seq = z_proj.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        out, _ = self.decoder_lstm(z_seq)                 # (B, T, H)
        recon = self.decoder_output(out)                   # (B, T, input_dim)
        return recon

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Returns dict with keys: recon, mu, logvar, z, health_pred
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        health_pred = self.health_head(mu).squeeze(-1)  # Use μ (not z) for deterministic health estimate
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "health_pred": health_pred,
        }


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = reconstruction loss + β × KL divergence.

    Returns (total_loss, recon_loss, kl_loss).
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


class BetaScheduler:
    """KL annealing: β increases linearly from 0 to 1 over warmup epochs."""

    def __init__(self, warmup_epochs: int = 30, max_beta: float = 1.0):
        self.warmup_epochs = warmup_epochs
        self.max_beta = max_beta

    def get_beta(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.max_beta
        return self.max_beta * (epoch / self.warmup_epochs)


def detect_anomalies(
    model: VAE_LSTM,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    threshold_sigma: float = 3.0,
) -> tuple[list[bool], list[float]]:
    """Flag cycles with reconstruction error > threshold_sigma × σ.

    Returns:
    - anomaly_flags: list of bool per sample
    - recon_errors: list of float (MSE per sample)
    """
    model.eval()
    all_errors = []
    with torch.no_grad():
        for xb, *_ in dataloader:
            xb = xb.to(device)
            out = model(xb)
            mse = F.mse_loss(out["recon"], xb, reduction="none").mean(dim=(1, 2))
            all_errors.extend(mse.cpu().tolist())

    errors = torch.tensor(all_errors)
    mu_err = errors.mean()
    std_err = errors.std()
    threshold = mu_err + threshold_sigma * std_err
    flags = (errors > threshold).tolist()
    return flags, all_errors


def train_vae(
    model: VAE_LSTM,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    y_train_health: torch.Tensor | None = None,
    *,
    max_epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 20,
    device: str = "cpu",
    warmup_epochs: int = 30,
    health_weight: float = 1.0,
) -> dict:
    """Train VAE-LSTM with KL annealing and optional health prediction loss."""
    from src.models.deep.lstm import EarlyStopping

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    beta_scheduler = BetaScheduler(warmup_epochs)
    early_stop = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        beta = beta_scheduler.get_beta(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 2:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
            else:
                xb = batch[0].to(device)
                yb = None

            optimizer.zero_grad()
            out = model(xb)
            total, recon_l, kl_l = vae_loss(xb, out["recon"], out["mu"], out["logvar"], beta)

            if yb is not None:
                health_loss = F.l1_loss(out["health_pred"], yb)
                total = total + health_weight * health_loss

            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                else:
                    xb = batch[0].to(device)
                    yb = None
                out = model(xb)
                total, _, _ = vae_loss(xb, out["recon"], out["mu"], out["logvar"], beta)
                if yb is not None:
                    total = total + health_weight * F.l1_loss(out["health_pred"], yb)
                val_loss += total.item()
                n_val += 1

        val_losses.append(val_loss / max(n_val, 1))
        scheduler.step(val_losses[-1])

        if early_stop.step(val_losses[-1], model):
            break

    early_stop.load_best(model)
    return {"train_losses": train_losses, "val_losses": val_losses}
