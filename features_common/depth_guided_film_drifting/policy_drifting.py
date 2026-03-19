"""
DA3-FiLM + Drifting Head Policy
=================================

Replaces the DDPM head with a one-step Drifting Model generator based on:
  "Generative Modeling via Drifting" (arXiv:2602.04770)

Key changes vs DA3Film2ModelPolicy (DDPM version):
  - No DDPMScheduler, no timestep sampling, no iterative denoising
  - Training objective: drifting loss (Eq. 6 — MSE toward drifted target)
  - Cross-coefficient normalization to preserve anti-symmetry (Prop. 3.1)
  - Inference: 1 NFE (vs 100 for DDPM) → ~100× faster
  - Same ConditionalUnet1D architecture, just with fixed timestep=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


# ============================================================
# Drifting Field Utilities  (arXiv:2602.04770)
#
# Cross-coefficient normalization (联合归一化):
#   w_pos and w_neg share the SAME denominator Z to preserve
#   anti-symmetry V_{p,q} = -V_{q,p}, guaranteeing V=0 at
#   equilibrium (gen distribution = data distribution).
#
#   Z[i] = Σ_j k(x_i, pos_j) + Σ_{j≠i} k(x_i, gen_j)
#
# This is critical — using separate softmax breaks anti-symmetry
# (see paper Table 1: FID degrades from 8.46 to 40+).
# ============================================================

def compute_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
) -> torch.Tensor:
    """
    Compute drift field V — based on official toy_mean_drift.py,
    with adaptive temperature for high-dimensional action spaces.

    In the official code (D≈4-8 in latent space), temp=0.05 works directly.
    For raw action space (D=112), distances are ~15, so we normalize
    distances by median_dist (following paper A.6 feature normalization)
    before applying temperature.

    Args:
        gen:   [G, D] generated samples
        pos:   [P, D] data samples
        temp:  temperature applied after distance normalization

    Returns:
        V: [G, D] drift vectors
    """
    G = gen.shape[0]

    # Concatenate targets: [G+P, D]
    targets = torch.cat([gen, pos], dim=0)

    # Unified distance matrix: [G, G+P]
    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self in gen-gen block

    # Adaptive normalization for high-dim: normalize distances by median
    # (paper A.6: normalize features so avg pairwise dist ≈ √C_j)
    median_dist = dist[:, G:].median().clamp(min=1e-6)
    dist_normalized = dist / median_dist  # now median dist ≈ 1.0

    # Unnormalized kernel (with normalized distances)
    kernel = (-dist_normalized / temp).exp()  # [G, G+P]

    # Bi-axial normalization: sqrt(row_sum * col_sum)
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer  # [G, G+P]

    # Cross-attention coefficients (official formula)
    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]  # [G, D]

    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]  # [G, D]

    return pos_V - neg_V


def drifting_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
    episode_ids: torch.Tensor = None,
) -> torch.Tensor:
    """
    Drifting training loss (Paper Eq. 6):
      L = MSE( x, stopgrad(x + V) )

    When episode_ids is provided, computes drift independently per episode
    (paper Section 4: "For each label, we perform Alg. 1 independently").

    Args:
        gen:         [B, D] generated samples
        pos:         [B, D] data samples
        temp:        temperature
        episode_ids: [B] int tensor, episode index per sample (or None for global)

    Returns:
        scalar loss
    """
    if episode_ids is None:
        # Fallback: global drift (original behavior)
        with torch.no_grad():
            V = compute_drift(gen, pos, temp=temp)
            target = (gen + V).detach()
        return F.mse_loss(gen, target)

    # Per-episode grouped drift
    unique_eps = episode_ids.unique()
    total_loss = 0.0
    total_count = 0
    for ep in unique_eps:
        mask = episode_ids == ep
        n_ep = mask.sum().item()
        if n_ep < 4:  # skip tiny groups
            continue
        gen_ep = gen[mask]
        pos_ep = pos[mask]
        with torch.no_grad():
            V_ep = compute_drift(gen_ep, pos_ep, temp=temp)
            target_ep = (gen_ep + V_ep).detach()
        total_loss = total_loss + F.mse_loss(gen_ep, target_ep) * n_ep
        total_count += n_ep

    if total_count == 0:
        return torch.tensor(0.0, device=gen.device, requires_grad=True)
    return total_loss / total_count


def drifting_loss_per_step(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
    episode_ids: torch.Tensor = None,
) -> torch.Tensor:
    """
    Per-timestep drifting loss: compute drift independently for each
    time step t in {0, ..., H-1} in action_dim space (D=14),
    avoiding the dimensionality curse of D=H*action_dim=112.

    In D=14, the kernel exp(-dist/temp) has healthy values (similar to
    the paper's toy examples at D=4-8), so the drift field V is non-zero
    and provides meaningful gradients.

    Args:
        gen:         [B, H, D] generated actions (NOT flattened)
        pos:         [B, H, D] data actions (NOT flattened)
        temp:        temperature
        episode_ids: [B] int tensor, episode index per sample (or None)

    Returns:
        scalar loss (averaged over timesteps and samples)
    """
    B, H, D = gen.shape
    total_loss = 0.0

    for t in range(H):
        gen_t = gen[:, t, :]   # [B, D=14]
        pos_t = pos[:, t, :]   # [B, D=14]

        if episode_ids is None:
            with torch.no_grad():
                V_t = compute_drift(gen_t, pos_t, temp=temp)
                target_t = (gen_t + V_t).detach()
            total_loss = total_loss + F.mse_loss(gen_t, target_t)
        else:
            unique_eps = episode_ids.unique()
            step_loss = 0.0
            step_count = 0
            for ep in unique_eps:
                mask = episode_ids == ep
                n_ep = mask.sum().item()
                if n_ep < 4:
                    continue
                gen_ep = gen_t[mask]
                pos_ep = pos_t[mask]
                with torch.no_grad():
                    V_ep = compute_drift(gen_ep, pos_ep, temp=temp)
                    target_ep = (gen_ep + V_ep).detach()
                step_loss = step_loss + F.mse_loss(gen_ep, target_ep) * n_ep
                step_count += n_ep
            if step_count > 0:
                total_loss = total_loss + step_loss / step_count

    return total_loss / H

def compute_local_drift(gen_k: torch.Tensor, pos: torch.Tensor, temp: float = 1.0):
    """
    计算条件生成下的局部漂移场 (Per-Observation Drift / Multi-Positive Drift)

    采用与 compute_drift 相同的联合归一化 (cross-coefficient normalization)，
    将 positive expert 动作集合作为额外 target 纳入统一的距离矩阵和 kernel 中，
    以保证 Proposition 3.1 要求的反对称性 V_{p,q} = -V_{q,p}。

    不做联合归一化时，pos_V 权重恒为 1 而 neg_V 分布在 K 个样本上，
    导致 target = expert + K/(K-1)*(gen-mean) 产生放大因子→训练发散。

    Args:
        gen_k: [B, K, D] (B个独立观测，每个观测生成K个动作)
        pos:   [B, D] 或 [B, P, D] (B个观测对应的1个或P个专家真实动作)
        temp:  温度超参数 (建议 1.0)
    """
    if pos.ndim == 2:
        pos = pos.unsqueeze(1)
    if pos.ndim != 3:
        raise ValueError(f"Expected pos to have shape [B, D] or [B, P, D], got {tuple(pos.shape)}")

    B, K, D = gen_k.shape
    if pos.shape[0] != B or pos.shape[-1] != D:
        raise ValueError(
            f"Shape mismatch: gen_k={tuple(gen_k.shape)}, pos={tuple(pos.shape)}"
        )
    # 将 positive 集合作为额外 target: targets = [gen_k | pos]
    # targets: [B, K+P, D]
    targets = torch.cat([gen_k, pos], dim=1)  # [B, K+P, D]

    # 统一距离矩阵: gen_k vs targets (含自身和 positive 集合)
    # dist: [B, K, K+P]
    dist = torch.cdist(gen_k, targets)

    # 屏蔽自身距离 (gen_k[b,i] vs targets[b,i] for i < K) — 向量化
    idx = torch.arange(K, device=dist.device)
    dist[:, idx, idx] = 1e6

    # 自适应距离归一化: 用 gen→pos 的中位距离
    # (paper A.6: normalize distances so median ≈ 1.0)
    dist_gen_pos = dist[:, :, K:]  # [B, K, P] — gen 到 positive 集合的距离
    median_dist = dist_gen_pos.median().detach().clamp(min=1e-6)
    dist_normalized = dist / median_dist

    # 未归一化的核
    kernel = (-dist_normalized / temp).exp()  # [B, K, K+P]

    # 双轴归一化 (Bi-axial normalization)
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    W = kernel / normalizer  # [B, K, K+P]

    # Cross-coefficient (与 compute_drift 相同的公式):
    # pos_coeff = W[:,:,K:] * W[:,:,:K].sum(dim=-1, keepdim=True)
    # neg_coeff = W[:,:,:K] * W[:,:,K:].sum(dim=-1, keepdim=True)
    W_pos = W[:, :, K:]   # [B, K, P] — gen→positive 集合的归一化核
    W_neg = W[:, :, :K]   # [B, K, K] — gen→gen 的归一化核

    # pos_V: 吸引力 (weighted attraction toward positive set)
    pos_coeff = W_pos * W_neg.sum(dim=-1, keepdim=True)   # [B, K, P]
    pos_V = torch.bmm(pos_coeff, pos)                     # [B, K, D]

    # neg_V: 排斥力 (weighted repulsion from other gen samples)
    neg_coeff = W_neg * W_pos.sum(dim=-1, keepdim=True)    # [B, K, K]
    neg_V = torch.bmm(neg_coeff, gen_k)                    # [B, K, D]

    # 总漂移场 (不含 K 补偿，由外部 drift_scale 控制放大倍数)
    V = pos_V - neg_V  # [B, K, D]
    return V

# ============================================================
# Drifting Policy (drop-in replacement for DA3Film2ModelPolicy)
# ============================================================

class DA3FilmDriftingPolicy(nn.Module):
    """
    2-model DA3-FiLM encoder + Drifting head.

    Identical to DA3Film2ModelPolicy except:
      - DDPM loss replaced by drifting loss
      - No DDPMScheduler, no timestep sampling
      - Inference: single forward pass (1 NFE) with fixed timestep=0
      - Same ConditionalUnet1D architecture
    """

    def __init__(
        self,
        fusion_encoder,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
        drifting_temp_scale: float = 0.05,
        drift_normalize: bool = False,
        drift_norm_mode: str = "per_obs",
        drift_norm_eps: float = 1e-6,
        drift_norm_ema_decay: float = 0.99,
        drift_target_rms: float = 1.0,
        drift_use_scale: bool = True,
        bc_mode: str = "fixed",
    ):
        super().__init__()
        self.fusion_encoder = fusion_encoder
        self.proprio_dim    = proprio_dim
        self.action_dim     = action_dim
        self.horizon        = horizon
        self.n_obs_steps    = n_obs_steps
        self.n_action_steps = n_action_steps
        self.drifting_temp_scale = drifting_temp_scale
        self.drift_normalize = drift_normalize
        self.drift_norm_mode = drift_norm_mode
        self.drift_norm_eps = drift_norm_eps
        self.drift_norm_ema_decay = drift_norm_ema_decay
        self.drift_target_rms = drift_target_rms
        self.drift_use_scale = drift_use_scale
        self.drift_scale = 1.0  # default: no scaling (overridden externally)
        self.bc_lambda = 0.0   # BC anchor weight (0 = pure drifting, >0 = hybrid)
        self.bc_mode = bc_mode
        # Running RMS for EMA normalization mode.
        # 0 means "not initialized yet".
        self.register_buffer("drift_norm_rms_ema", torch.tensor(0.0))
        # Training diagnostics (last forward values)
        self.register_buffer("drift_last_raw_rms", torch.tensor(0.0))
        self.register_buffer("drift_last_norm_rms", torch.tensor(0.0))
        self.register_buffer("drift_last_step_scale", torch.tensor(1.0))
        self.register_buffer("bc_last_weight", torch.tensor(0.0))
        self.register_buffer("bc_last_pos_spread", torch.tensor(0.0))

        from diffusion_policy.model.common.normalizer import LinearNormalizer
        self.normalizer = LinearNormalizer()

        per_step_dim    = fusion_encoder.out_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Same ConditionalUnet1D as DDPM version (DA3Film2ModelPolicy)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

    def _normalize_drift(self, V: torch.Tensor) -> torch.Tensor:
        if (not self.drift_normalize) or self.drift_norm_mode == "none":
            if self.training:
                with torch.no_grad():
                    raw_rms = V.pow(2).mean().sqrt()
                    self.drift_last_raw_rms.copy_(raw_rms.detach())
                    self.drift_last_norm_rms.copy_(raw_rms.detach())
            return V

        D = max(V.shape[-1], 1)
        with torch.no_grad():
            raw_rms = V.pow(2).mean().sqrt()
            if self.training:
                self.drift_last_raw_rms.copy_(raw_rms.detach())

        if self.drift_norm_mode == "batch":
            scale = V.pow(2).sum(dim=-1).mean().div(D).sqrt()
        elif self.drift_norm_mode == "per_obs":
            scale = V.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).div(D).sqrt()
            scale = scale.unsqueeze(-1)
        elif self.drift_norm_mode == "ema":
            # Normalize by running RMS (EMA) instead of current batch RMS.
            # This avoids forcing MSE to an exact constant each iteration.
            batch_rms = V.pow(2).mean().sqrt().detach()
            if self.training:
                if float(self.drift_norm_rms_ema.item()) <= 0.0:
                    self.drift_norm_rms_ema.copy_(batch_rms)
                else:
                    self.drift_norm_rms_ema.mul_(self.drift_norm_ema_decay).add_(
                        batch_rms, alpha=1.0 - self.drift_norm_ema_decay
                    )
            if float(self.drift_norm_rms_ema.item()) <= 0.0:
                scale = batch_rms
            else:
                scale = self.drift_norm_rms_ema
        else:
            raise ValueError(f"Unsupported drift_norm_mode: {self.drift_norm_mode}")

        V_norm = V / scale.clamp_min(self.drift_norm_eps)
        # Optional amplitude calibration:
        # keep direction from drifting field, but set target RMS to a task-friendly value.
        # This is equivalent to a principled step-size in high-dimensional action space.
        if self.drift_normalize:
            V_norm = V_norm * float(self.drift_target_rms)
        if self.training:
            with torch.no_grad():
                self.drift_last_norm_rms.copy_(V_norm.pow(2).mean().sqrt().detach())
        return V_norm

    def _effective_drift_scale(self) -> float:
        return float(self.drift_scale) if self.drift_use_scale else 1.0

    def compute_loss(
        self,
        tokens_list,
        agent_pos,
        actions,
        K=8,
        sub_K=None,
        positive_actions=None,
        **kwargs,
    ):
        """
        K-sampling Drifting Loss (论文 Section 4 条件生成).

        对每个观测生成 K 个样本，在同一观测内计算局部漂移场：
          - 吸引力：K 个生成样本被 1 个或多个专家动作吸引
          - 排斥力：K 个生成样本内部互相排斥
        视觉编码器只运行 B 次。

        当 sub_K < K 时使用 Multi-Sub 优化:
          - 所有 K 个粒子先通过 UNet (no_grad) 计算漂移场 → 高质量漂移
          - 仅 sub_K 个粒子重新通过 UNet (with grad) 计算损失 → 低显存

        tokens_list: List[2 × Tensor [B, To, K_i, C_i]]
        agent_pos:   [B, To, 14]
        actions:     [B, horizon, 14]
        K:           每个观测的漂移粒子总数
        sub_K:       梯度回传的粒子子集 (None = 全部 K)
        """
        B = actions.shape[0]
        device = actions.device
        D = self.horizon * self.action_dim  # 展平维度 (8×14=112)

        nactions = self.normalizer.normalize({"action": actions})["action"].to(device)
        nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)

        # Encode observations (视觉编码只做 1 次)
        fused = self.fusion_encoder(tokens_list)
        obs_cond = self.obs_encoder(
            torch.cat([fused, nagent_pos], dim=-1).reshape(B, -1)
        )  # [B, 256]

        if positive_actions is None:
            pos_flat = nactions.reshape(B, 1, D)    # [B, 1, 112]
        else:
            if positive_actions.ndim == 3:
                positive_actions = positive_actions.unsqueeze(1)
            npositive = self.normalizer.normalize({"action": positive_actions})["action"].to(device)
            pos_flat = npositive.reshape(B, npositive.shape[1], D)  # [B, P, 112]

        if sub_K is None or sub_K >= K:
            # === 标准路径: 全部 K 粒子 with grad ===
            obs_cond_k = obs_cond.repeat_interleave(K, dim=0)
            noise_k = torch.randn((B * K, self.horizon, self.action_dim), device=device)
            timestep_k = torch.zeros(B * K, dtype=torch.long, device=device)

            gen_k = self.noise_pred_net(noise_k, timestep_k, global_cond=obs_cond_k)
            gen_flat_k = gen_k.reshape(B, K, D)

            with torch.no_grad():
                V = compute_local_drift(gen_flat_k, pos_flat, temp=self.drifting_temp_scale)
                V = self._normalize_drift(V)
                step_scale = self._effective_drift_scale()
                if self.training:
                    self.drift_last_step_scale.fill_(step_scale)
                target_k = (gen_flat_k + step_scale * V).detach()

            loss = F.mse_loss(gen_flat_k, target_k)

            # Hybrid BC anchor: small MSE toward expert to stabilize late-epoch
            if self.bc_lambda > 0.0:
                gen_mean = gen_flat_k.mean(dim=1)           # [B, D]
                pos_mean = pos_flat.mean(dim=1)             # [B, D]
                bc_loss_per = F.mse_loss(
                    gen_mean, pos_mean.detach(), reduction="none"
                ).mean(dim=-1)
                if self.bc_mode == "fixed":
                    bc_weight = torch.full_like(bc_loss_per, float(self.bc_lambda))
                    pos_spread = torch.zeros_like(bc_loss_per)
                elif self.bc_mode == "diversity_aware":
                    pos_spread = (pos_flat - pos_mean.unsqueeze(1)).pow(2).mean(dim=-1).sqrt().mean(dim=-1)
                    bc_weight = float(self.bc_lambda) / (1.0 + pos_spread.detach())
                else:
                    raise ValueError(f"Unsupported bc_mode: {self.bc_mode}")

                bc_loss = (bc_weight * bc_loss_per).mean()
                loss = loss + bc_loss
                if self.training:
                    with torch.no_grad():
                        self.bc_last_weight.copy_(bc_weight.mean().detach())
                        self.bc_last_pos_spread.copy_(pos_spread.mean().detach())

            return loss

        # === Multi-Sub 路径: no_grad 全量 K 计算漂移, with grad sub_K 计算损失 ===
        noise_all = torch.randn((B * K, self.horizon, self.action_dim), device=device)
        timestep_all = torch.zeros(B * K, dtype=torch.long, device=device)
        obs_cond_all = obs_cond.repeat_interleave(K, dim=0)

        with torch.no_grad():
            gen_all = self.noise_pred_net(noise_all, timestep_all, global_cond=obs_cond_all)
            gen_flat_all = gen_all.reshape(B, K, D)
            V = compute_local_drift(gen_flat_all, pos_flat, temp=self.drifting_temp_scale)
            V = self._normalize_drift(V)
            step_scale = self._effective_drift_scale()
            if self.training:
                self.drift_last_step_scale.fill_(step_scale)
            target_all = (gen_flat_all + step_scale * V)  # [B, K, D]

        # 取最后 sub_K 个粒子重新过 UNet (with grad)
        sub_start = K - sub_K
        noise_sub = noise_all.reshape(B, K, self.horizon, self.action_dim)[:, sub_start:].reshape(
            B * sub_K, self.horizon, self.action_dim
        )
        obs_cond_sub = obs_cond.repeat_interleave(sub_K, dim=0)
        timestep_sub = torch.zeros(B * sub_K, dtype=torch.long, device=device)

        gen_sub = self.noise_pred_net(noise_sub, timestep_sub, global_cond=obs_cond_sub)
        gen_flat_sub = gen_sub.reshape(B, sub_K, D)

        target_sub = target_all[:, sub_start:].detach()  # [B, sub_K, D]
        loss = F.mse_loss(gen_flat_sub, target_sub)
        return loss

    @torch.no_grad()
    def predict_action(self, tokens_list, agent_pos=None, K_ensemble=1):
        """
        Single-step inference (1 NFE), optionally with K-ensemble.

        When K_ensemble > 1, generates K action candidates per observation
        and returns the coordinate-wise median (more robust than mean).
        Cost: K × single_forward ≈ K × 2.8ms, negligible vs backbone.

        Args:
            tokens_list: List of token tensors
            agent_pos: [B, n_obs_steps, proprio_dim] or None
            K_ensemble: number of parallel samples for ensemble (default=1)

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

        K = max(1, int(K_ensemble))
        if K == 1:
            # Single forward pass — fixed timestep=0, no iteration
            noise = torch.randn((B, self.horizon, self.action_dim), device=device)
            timestep = torch.zeros(B, dtype=torch.long, device=device)
            action = self.noise_pred_net(
                noise, timestep, global_cond=obs_cond
            )                                                  # [B, H, D]
        else:
            # K-ensemble: generate K candidates, take coordinate-wise median
            obs_cond_k = obs_cond.repeat_interleave(K, dim=0)  # [B*K, 256]
            noise = torch.randn((B * K, self.horizon, self.action_dim), device=device)
            timestep = torch.zeros(B * K, dtype=torch.long, device=device)
            action_k = self.noise_pred_net(
                noise, timestep, global_cond=obs_cond_k
            )                                                  # [B*K, H, D]
            action_k = action_k.reshape(B, K, self.horizon, self.action_dim)
            action = action_k.median(dim=1).values             # [B, H, D]

        return self.normalizer.unnormalize({"action": action})["action"]
