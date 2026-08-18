from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TrajectoryMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 96, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualIntentTrajectoryMLP(nn.Module):
    """Passive state forecast plus a residual conditioned on the action query."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 96, dropout: float = 0.05, action_dim: int = 2):
        super().__init__()
        state_dim = input_dim - action_dim
        self.state_dim = state_dim
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.intent = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.state_dim]
        return self.base(state) + self.intent(x)


class LightweightGraphIntentPredictor(nn.Module):
    """Small graph message-passing predictor with the same flat ICGP I/O contract.

    The feature layout follows ``src.dataset.make_feature``:
    ego context (8), up to four neighbor rows (4 x 6), and candidate action (2).
    This keeps the ablation isolated to the predictor architecture; the planner,
    candidate set, safety screen, and RVO-style support layer are unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        ego_dim: int = 8,
        neighbor_limit: int = 4,
        neighbor_dim: int = 6,
        action_dim: int = 2,
    ):
        super().__init__()
        expected_dim = ego_dim + neighbor_limit * neighbor_dim + action_dim
        if input_dim != expected_dim:
            raise ValueError(f"Expected input_dim={expected_dim}, got {input_dim}")
        if output_dim % (neighbor_limit * 2) != 0:
            raise ValueError("output_dim must encode H x K x 2 neighbor positions")
        self.ego_dim = ego_dim
        self.neighbor_limit = neighbor_limit
        self.neighbor_dim = neighbor_dim
        self.action_dim = action_dim
        self.horizon = output_dim // (neighbor_limit * 2)

        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(neighbor_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ego = x[:, : self.ego_dim]
        neighbor_start = self.ego_dim
        neighbor_end = neighbor_start + self.neighbor_limit * self.neighbor_dim
        neighbors = x[:, neighbor_start:neighbor_end].reshape(-1, self.neighbor_limit, self.neighbor_dim)
        action = x[:, -self.action_dim :]

        action_nodes = action[:, None, :].expand(-1, self.neighbor_limit, -1)
        ego_h = self.ego_encoder(torch.cat([ego, action], dim=-1))
        node_h = self.node_encoder(torch.cat([neighbors, action_nodes], dim=-1))

        mask = torch.linalg.norm(neighbors, dim=-1) > 1.0e-6
        q = self.query(node_h)
        k = self.key(node_h)
        v = self.value(node_h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        key_mask = mask[:, None, :].expand_as(scores)
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        msg = torch.matmul(attn, v)
        node_h = self.message_norm(node_h + self.dropout(msg))

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(node_h.dtype)
        graph_h = (node_h * mask[:, :, None].to(node_h.dtype)).sum(dim=1) / denom
        graph_h = self.fusion(torch.cat([ego_h, graph_h], dim=-1))
        graph_nodes = graph_h[:, None, :].expand(-1, self.neighbor_limit, -1)
        out = self.node_head(torch.cat([node_h, graph_nodes], dim=-1))
        return out.reshape(x.shape[0], self.neighbor_limit, self.horizon, 2).permute(0, 2, 1, 3).reshape(
            x.shape[0], self.horizon * self.neighbor_limit * 2
        )


class ConditionalTrajectoryVAE(nn.Module):
    """Conditional VAE for multi-modal short-horizon trajectory prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.cond = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cond(x)
        h = self.encoder(torch.cat([c, y], dim=-1))
        return c, self.mu(h), self.logvar(h).clamp(-8.0, 6.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, x: torch.Tensor, z: torch.Tensor | None = None, n_samples: int = 1) -> torch.Tensor:
        c = self.cond(x)
        if z is None:
            if n_samples == 1:
                z = torch.zeros((x.shape[0], self.latent_dim), device=x.device, dtype=x.dtype)
                return self.decoder(torch.cat([c, z], dim=-1))
            z = torch.randn((n_samples, x.shape[0], self.latent_dim), device=x.device, dtype=x.dtype)
            c_rep = c.unsqueeze(0).expand(n_samples, -1, -1)
            out = self.decoder(torch.cat([c_rep, z], dim=-1).reshape(n_samples * x.shape[0], -1))
            return out.reshape(n_samples, x.shape[0], self.output_dim)
        return self.decoder(torch.cat([c, z], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c, mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(torch.cat([c, z], dim=-1))
        return recon, mu, logvar
