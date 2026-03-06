"""
DA3-FiLM + Drifting Head Policy
=================================

Replaces the DDPM head with a one-step Drifting Model generator based on:
  "Generative Modeling via Drifting" (arXiv:2602.04770), He et al. 2026

Key changes vs DA3Film2ModelPolicy (DDPM version):
  - No DDPMScheduler, no ConditionalUnet1D, no timestep embedding
  - Generator: noise [B, H, D] + obs_cond [B, 256] → action [B, H, D]  (single forward pass)
  - Training objective: drifting loss (mean-shift based MSE toward drifted target)
  - Inference: 1 NFE (vs 100 for DDPM) → ~100× faster

Architecture:
  DriftingActionGenerator: 4-layer MLP with SiLU activation
    Input:  flatten(noise) [B, H*D]  +  obs_cond [B, obs_cond_dim]
    Output: reshape to [B, H, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Drifting Field Utilities
# ============================================================

def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """
    Compute batch-normalized mean-shift drifting field V(gen).

    Implements the compact formula from arXiv:2602.04770:
        V(x) = kernel_weighted_mean(pos) - kernel_weighted_mean(gen)
    with the kernel normalized over the combined mini-batch (gen + pos).

    IMPORTANT: temp must be scaled to the action space dimensionality.
      - Original Colab 2D toy: temp=0.05  (L2 distances ≈ 0.5)
      - 112D action space (H=8, D=14): temp≈1.0  (L2 distances ≈ √112 ≈ 7.5)
      Rule of thumb: temp ≈ 0.05 × √(dim / 2)

    Args:
        gen:  [G, D] generated samples (flattened action sequences)
        pos:  [P, D] real data samples (flattened action sequences)
        temp: temperature for softmax kernel (must be calibrated to data scale)

    Returns:
        V:    [G, D] drift vectors; V ≈ 0 when gen ≈ pos
    """
    G = gen.shape[0]
    targets = torch.cat([gen, pos], dim=0)    # [G+P, D]
    dist = torch.cdist(gen, targets)           # [G, G+P]

    # Batch-normalized kernel: softmax over all targets (gen + pos combined)
    w = F.softmax(-dist / temp, dim=-1)        # [G, G+P], rows sum to 1

    w_gen = w[:, :G]   # [G, G]  — similarity to generated samples
    w_pos = w[:, G:]   # [G, P]  — similarity to real data

    mean_pos = w_pos @ pos   # [G, D]  pull toward real data
    mean_gen = w_gen @ gen   # [G, D]  push away from generated

    return mean_pos - mean_gen  # [G, D]


def drifting_loss(gen: torch.Tensor, pos: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """
    Drifting training loss: MSE( gen, stopgrad(gen + V) )

    The generator is trained to reduce the distance to its own drifted version.
    At equilibrium (gen ≈ pos), V ≈ 0 and loss ≈ 0.

    Args:
        gen:  [B, D] generated samples (flattened, normalized)
        pos:  [B, D] real data samples  (flattened, normalized)
        temp: drifting field temperature

    Returns:
        scalar loss
    """
    with torch.no_grad():
        V = compute_drift(gen, pos, temp=temp)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


# ============================================================
# Direct Action Generator (replaces ConditionalUnet1D)
# ============================================================

class DriftingActionGenerator(nn.Module):
    """
    Single-forward-pass action generator for Drifting Model.

    Maps: (noise, obs_cond) → action
    in a single forward pass (1 NFE), replacing 100-step DDPM denoising.

    Architecture: 4-layer MLP with SiLU activation.
    Input:  flatten(noise) [B, horizon*action_dim]  || obs_cond [B, obs_cond_dim]
    Output: [B, horizon, action_dim]
    """

    def __init__(
        self,
        action_dim: int = 14,
        horizon: int = 8,
        obs_cond_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.action_dim   = action_dim
        self.horizon      = horizon
        noise_flat_dim    = action_dim * horizon
        input_dim         = noise_flat_dim + obs_cond_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, noise_flat_dim),
        )
        # Use default Kaiming initialization — do NOT override with near-zero.
        # Near-zero init causes gen ≈ 0 (all identical), which collapses the
        # batch-normalized kernel (V ≈ 0) and produces zero loss with no gradient.

    def forward(self, noise: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise:    [B, horizon, action_dim] Gaussian noise
            obs_cond: [B, obs_cond_dim]
        Returns:
            action:   [B, horizon, action_dim]  =  noise + correction(noise, obs)

        Residual design: gen = noise + f_theta(noise, obs).
          - At init: correction ≈ 0 → gen ≈ noise (std ≈ 1, highly diverse)
          - This ensures the drifting kernel always has meaningful variance,
            avoiding the degenerate V ≈ 0 case that arises when gen ≈ 0.
          - At convergence: gen = noise + correction that maps noise → real actions.
        """
        B = noise.shape[0]
        noise_flat = noise.reshape(B, -1)                          # [B, H*D]
        x = torch.cat([noise_flat, obs_cond], dim=-1)              # [B, H*D+obs]
        correction = self.net(x)                                   # [B, H*D]
        action_flat = noise_flat + correction                      # residual: noise + Δ
        return action_flat.reshape(B, self.horizon, self.action_dim)


# ============================================================
# Drifting Policy (drop-in replacement for DA3Film2ModelPolicy)
# ============================================================

class DA3FilmDriftingPolicy(nn.Module):
    """
    2-model DA3-FiLM encoder + Drifting head.

    Identical to DA3Film2ModelPolicy except:
      - DDPM head (ConditionalUnet1D + DDPMScheduler) is replaced by
        DriftingActionGenerator + drifting_loss
      - Inference: single forward pass (1 NFE)
    """

    def __init__(
        self,
        fusion_encoder,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
        drifting_temp: float = 3.0,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.fusion_encoder = fusion_encoder
        self.proprio_dim    = proprio_dim
        self.action_dim     = action_dim
        self.horizon        = horizon
        self.n_obs_steps    = n_obs_steps
        self.n_action_steps = n_action_steps
        self.drifting_temp  = drifting_temp

        # Lazy import: ensure DP path is on sys.path before this
        # (the train/deploy scripts set it up; policy_drifting.py itself is path-agnostic)
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        self.normalizer = LinearNormalizer()

        per_step_dim    = fusion_encoder.out_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps

        # Observation MLP encoder (same as DDPM version)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # ★ Drifting generator (replaces ConditionalUnet1D + DDPMScheduler)
        self.action_generator = DriftingActionGenerator(
            action_dim=action_dim,
            horizon=horizon,
            obs_cond_dim=256,
            hidden_dim=hidden_dim,
        )

    def _encode_obs(self, tokens_list, agent_pos):
        """Shared observation encoding (used by both training and inference)."""
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        fused = self.fusion_encoder(tokens_list)     # [B, To, out_dim]
        nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
        obs_combined = torch.cat([fused, nagent_pos], dim=-1)    # [B, To, out_dim+proprio]
        obs_cond = self.obs_encoder(obs_combined.reshape(B, -1))  # [B, 256]
        return obs_cond

    def compute_loss(self, tokens_list, agent_pos, actions):
        """
        Drifting training loss.

        tokens_list: List[2 × Tensor [B, To, K_i, C_i]]
        agent_pos:   [B, To, 14]
        actions:     [B, horizon, 14]
        """
        B = actions.shape[0]
        device = actions.device
        nactions   = self.normalizer.normalize({"action": actions})["action"].to(device)
        nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)

        # Encode observations
        fused = self.fusion_encoder(tokens_list)           # [B, To, out_dim]
        obs_cond = self.obs_encoder(
            torch.cat([fused, nagent_pos], dim=-1).reshape(B, -1)
        )

        # Generate actions from noise
        noise = torch.randn_like(nactions)
        gen   = self.action_generator(noise, obs_cond)     # [B, H, D]

        # Drifting loss in flattened action space
        gen_flat = gen.reshape(B, -1)                      # [B, H*D]
        pos_flat = nactions.reshape(B, -1)                 # [B, H*D]
        loss = drifting_loss(gen_flat, pos_flat, temp=self.drifting_temp)
        return loss

    @torch.no_grad()
    def predict_action(self, tokens_list, agent_pos=None):
        """
        Single-step inference (1 NFE).

        Returns: [B, horizon, action_dim] unnormalized actions
        """
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        fused = self.fusion_encoder(tokens_list)

        if agent_pos is not None:
            nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            zeros = torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)
            obs_combined = torch.cat([fused, zeros], dim=-1)

        obs_cond = self.obs_encoder(obs_combined.reshape(B, -1))

        # Single forward pass — no iteration!
        noise  = torch.randn((B, self.horizon, self.action_dim), device=device)
        action = self.action_generator(noise, obs_cond)    # [B, H, D]

        return self.normalizer.unnormalize({"action": action})["action"]
