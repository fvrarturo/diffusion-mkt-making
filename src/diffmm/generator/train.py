"""Lightning module for fine-tuning the conditional DDPM.

Implements Algorithm 1 from Ho et al. (2020) plus classifier-free-guidance
dropout per Ho & Salimans (2022). Phase B adds:

  - **v-parameterization** (Salimans & Ho 2022): the model predicts
    v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0 instead of ε. Trained as MSE between
    model output and v_true. Documented motivation: ε-prediction undertrains
    heavy tails on densities with rare extreme events (Work6_update §5).
  - **Conditioning curriculum**: epochs 0..ramp_start are fully unconditional
    (cfg_dropout = 1.0), then dropout linearly ramps from 1.0 → final value
    over [ramp_start, ramp_end]. Forces the unconditional pathway to
    establish a solid baseline before FiLM is asked to specialize, addressing
    the FiLM-collapse failure documented in Work5 §F.2 / Work6_update §5.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import pytorch_lightning as pl
    LightningModule = pl.LightningModule
except ImportError:  # allow import in environments without lightning
    class LightningModule(torch.nn.Module):
        def log(self, *a, **k): pass

from .model import GeneratorModel
from .sample import DDIMSchedule


@dataclass
class CFGDropoutCurriculum:
    """Linear ramp of cfg_dropout from `initial` to `final` over an epoch range.

    Set `enabled=False` (or leave the trainer's curriculum=None) to disable
    the curriculum and use a constant `cfg_dropout` value, matching v2/v3
    behavior.
    """
    enabled: bool = True
    initial: float = 1.0          # before ramp_start: fully unconditional
    final: float = 0.2            # after ramp_end: steady-state CFG dropout
    ramp_start_epoch: int = 2
    ramp_end_epoch: int = 5

    def value_at(self, epoch: int) -> float:
        if not self.enabled:
            return self.final
        if epoch < self.ramp_start_epoch:
            return self.initial
        if epoch >= self.ramp_end_epoch:
            return self.final
        # Linear interpolation between (ramp_start, initial) and (ramp_end, final).
        span = max(1, self.ramp_end_epoch - self.ramp_start_epoch)
        frac = (epoch - self.ramp_start_epoch) / span
        return float(self.initial + frac * (self.final - self.initial))


def _ramp(epoch: int, cfg: dict) -> float:
    """Linear ramp of an auxiliary-loss coefficient. Returns 0 before
    `ramp_start_epoch`, `lambda_final` after `ramp_end_epoch`, linear
    interpolation in between. Phase F Model B helper.
    """
    lam_final = float(cfg.get("lambda_final", 0.0))
    s = int(cfg.get("ramp_start_epoch", 0))
    e = int(cfg.get("ramp_end_epoch", s))
    if epoch < s:
        return 0.0
    if epoch >= e:
        return lam_final
    frac = (epoch - s) / max(e - s, 1)
    return lam_final * frac


class DDPMTrainer(LightningModule):
    def __init__(
        self,
        generator: GeneratorModel,
        schedule: DDIMSchedule,
        learning_rate: float = 1e-4,
        cfg_dropout: float = 0.1,
        weight_decay: float = 0.0,
        freeze_backbone: bool = False,
        min_snr_gamma: float | None = 5.0,
        prediction_type: str | None = None,
        cfg_dropout_curriculum: CFGDropoutCurriculum | None = None,
        noise_mask: list[float] | None = None,
        loss_mask: list[float] | None = None,
        aux_loss_cfg: dict | None = None,
    ):
        """
        Args:
            min_snr_gamma: if not None, use Min-SNR loss weighting per
                Hang et al. 2023. For ε-prediction, weight = min(SNR, γ) / SNR
                (down-weights low-noise t where ε-loss is naturally small).
                For v-prediction, weight = min(SNR, γ) / (SNR + 1) — derived
                so that the effective weight on the underlying x_0-MSE matches
                ε-prediction's weighting (Salimans & Ho 2022 §3.2). γ=5 is
                the paper default and works well for both parameterizations.
            prediction_type: "eps" or "v". If None, falls back to
                schedule.prediction_type (the source of truth — the schedule
                is what gets passed to the sampler at inference time).
            cfg_dropout_curriculum: epoch-based ramp of cfg_dropout. When
                None or `enabled=False`, the trainer uses the constant
                `cfg_dropout` value (v2/v3 behavior). When enabled, the
                `cfg_dropout` constructor arg is ignored and the curriculum
                drives the per-epoch dropout rate.
            noise_mask: per-feature 0/1 mask of length n_features. Channels
                with mask=0 are NOT noised (passed through clean as conditioning).
                Used by Phase F Model A (cascade): set [0,0,0,0,0,0,0,1] to
                only diffuse mid_return given clean book features. None →
                all-ones (noise every channel — v2 behavior).
            loss_mask: per-feature 0/1 mask of length n_features. Channels
                with mask=0 contribute 0 to the loss. Should match noise_mask
                for cascade training (predict only the noised channel).
            aux_loss_cfg: Phase F Model B auxiliary-loss configuration dict.
                Keys (all optional): joint, kurt, tail. Each is itself a
                dict with `lambda_final`, `ramp_start_epoch`, `ramp_end_epoch`,
                plus type-specific fields. Computed on the single-step x̂_0
                approximation. None → no auxiliary losses (v2 behavior).
        """
        super().__init__()
        self.generator = generator
        self.lr = learning_rate
        self.cfg_dropout = cfg_dropout
        self.weight_decay = weight_decay
        self.min_snr_gamma = min_snr_gamma
        self.prediction_type = prediction_type or schedule.prediction_type
        if self.prediction_type not in ("eps", "v"):
            raise ValueError(
                f"prediction_type must be 'eps' or 'v', got {self.prediction_type!r}"
            )
        self.curriculum = cfg_dropout_curriculum
        self.aux_loss_cfg = aux_loss_cfg or {}

        if freeze_backbone:
            for p in self.generator.denoiser.parameters():
                p.requires_grad = False

        self.register_buffer("alphas_cumprod", schedule.alphas_cumprod)

        # Per-feature noise/loss masks. Default: all ones (v2 behavior — every
        # channel noised, every channel contributes to loss).
        if noise_mask is not None:
            nm = torch.tensor(list(noise_mask), dtype=torch.float32)
            self.register_buffer("noise_mask", nm.view(1, 1, -1))
        else:
            self.noise_mask = None
        if loss_mask is not None:
            lm = torch.tensor(list(loss_mask), dtype=torch.float32)
            self.register_buffer("loss_mask", lm.view(1, 1, -1))
        else:
            self.loss_mask = None

    def _current_cfg_dropout(self) -> float:
        """Effective cfg_dropout for this epoch — curriculum-driven if enabled."""
        if self.curriculum is None or not self.curriculum.enabled:
            return self.cfg_dropout
        epoch = int(getattr(self, "current_epoch", 0))
        return self.curriculum.value_at(epoch)

    def training_step(self, batch, batch_idx):
        x0, c = batch
        return self._step(x0, c, "train")

    def validation_step(self, batch, batch_idx):
        x0, c = batch
        return self._step(x0, c, "val")

    def _step(self, x0: torch.Tensor, c: torch.Tensor, stage: str) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device
        T = self.alphas_cumprod.shape[0]

        t = torch.randint(0, T, (B,), device=device)
        a_t = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        eps = torch.randn_like(x0)

        # Phase F Model A: noise_mask gates which channels get diffused.
        # Channels with mask=0 stay clean (passed through as conditioning).
        # x_t for those channels equals x_0 — no noise added at any t.
        if self.noise_mask is not None:
            nm = self.noise_mask  # (1, 1, F)
            x_t = nm * (torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps) + (1 - nm) * x0
        else:
            x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps

        # CFG dropout — curriculum-driven if enabled, else constant.
        active_dropout = self._current_cfg_dropout() if stage == "train" else 0.0
        if active_dropout > 0:
            drop = torch.rand(B, device=device) < active_dropout
            if drop.any():
                c_null = self.generator.regime_embed.null_condition(B, device)
                c = torch.where(drop.unsqueeze(-1), c_null, c)

        pred = self.generator.predict_noise(x_t, t, c)

        # Build the regression target based on prediction_type.
        if self.prediction_type == "eps":
            target = eps
        else:  # "v"
            # v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0
            target = torch.sqrt(a_t) * eps - torch.sqrt(1 - a_t) * x0

        # Per-sample MSE. With loss_mask, only masked channels contribute —
        # used by Phase F Model A cascade Stage 2 (predict only mid_return).
        if self.loss_mask is not None:
            lm = self.loss_mask  # (1, 1, F)
            sq = ((pred - target) ** 2) * lm
            denom = lm.sum() * x0.shape[0] * x0.shape[1] / x0.shape[0]  # mean over masked entries
            # Per-sample mean over (L * masked_F) elements
            per_sample_mse = sq.sum(dim=(1, 2)) / (lm.sum() * x0.shape[1])
        else:
            reduce_dims = tuple(range(1, target.dim()))
            per_sample_mse = ((pred - target) ** 2).mean(dim=reduce_dims)

        if self.min_snr_gamma is not None:
            a_t_scalar = self.alphas_cumprod[t]                # (B,)
            snr = a_t_scalar / (1.0 - a_t_scalar)
            if self.prediction_type == "eps":
                # Hang et al. 2023: weight = min(SNR, γ) / SNR.
                weight = torch.clamp(snr, max=self.min_snr_gamma) / snr
            else:  # "v"
                # Salimans & Ho 2022 §3.2: for v-prediction, the per-sample
                # MSE is already (SNR + 1)× the underlying x_0-MSE. To match
                # ε-prediction's effective weighting on x_0, divide by SNR+1
                # instead of SNR.
                weight = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)
            loss = (per_sample_mse * weight).mean()
        else:
            loss = per_sample_mse.mean()

        # Phase F Model B: auxiliary losses computed on the single-step x̂_0
        # approximation. Stop-gradient through schedule constants; gradient
        # flows back through `pred`. Each component is curriculum-ramped.
        if self.aux_loss_cfg and stage == "train":
            with torch.no_grad():
                a_t_scalar = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
                inv_sqrt_a = 1.0 / torch.sqrt(a_t_scalar + 1e-12)
                sqrt_one_minus_a = torch.sqrt(1.0 - a_t_scalar)
            # x̂_0 = (x_t - √(1-ᾱ) ε_pred) / √ᾱ   for ε-pred
            # For v-pred: x̂_0 = √ᾱ x_t - √(1-ᾱ) v_pred
            if self.prediction_type == "eps":
                x0_pred = inv_sqrt_a * (x_t - sqrt_one_minus_a * pred)
            else:  # "v"
                x0_pred = torch.sqrt(a_t_scalar) * x_t - sqrt_one_minus_a * pred

            epoch = int(getattr(self, "current_epoch", 0))
            aux_total = 0.0

            # 1. Joint OFI→return slope preservation
            j_cfg = self.aux_loss_cfg.get("joint", {}) or {}
            if j_cfg.get("enabled", False):
                lam = _ramp(epoch, j_cfg)
                if lam > 0:
                    bid_sz_idx = int(j_cfg.get("bid_sz_idx", 2))
                    ask_sz_idx = int(j_cfg.get("ask_sz_idx", 3))
                    return_idx = int(j_cfg.get("return_idx", 7))
                    target_slope = float(j_cfg.get("target_slope", 1.11e-4))
                    # OFI proxy at each event = bid_sz - ask_sz, on x̂_0 scale.
                    # Then take the regression slope of return on this proxy
                    # within each window (B, L) → one slope per window.
                    ofi = x0_pred[..., bid_sz_idx] - x0_pred[..., ask_sz_idx]  # (B, L)
                    ret = x0_pred[..., return_idx]                              # (B, L)
                    # Per-window OLS slope: cov(x, y) / var(x). Mean-centered.
                    ofi_c = ofi - ofi.mean(dim=1, keepdim=True)
                    ret_c = ret - ret.mean(dim=1, keepdim=True)
                    var = (ofi_c ** 2).mean(dim=1) + 1e-12
                    cov = (ofi_c * ret_c).mean(dim=1)
                    slope_per_window = cov / var
                    joint_loss = ((slope_per_window - target_slope) ** 2).mean()
                    loss = loss + lam * joint_loss
                    aux_total += float(joint_loss.detach().item()) * lam
                    if stage == "train":
                        self.log("train/aux_joint_loss", joint_loss, on_epoch=True)
                        self.log("train/aux_joint_lambda", lam, on_epoch=True)

            # 2. Kurtosis hinge — penalize only if kurt(x̂_0_return) < target
            k_cfg = self.aux_loss_cfg.get("kurt", {}) or {}
            if k_cfg.get("enabled", False):
                lam = _ramp(epoch, k_cfg)
                if lam > 0:
                    return_idx = int(k_cfg.get("return_idx", 7))
                    target_kurt = float(k_cfg.get("target_kurt", 500.0))
                    r = x0_pred[..., return_idx].reshape(-1)
                    r = r - r.mean()
                    var_r = (r ** 2).mean() + 1e-12
                    fourth = (r ** 4).mean()
                    kurt = fourth / (var_r ** 2) - 3.0  # excess kurtosis
                    deficit = torch.clamp(target_kurt - kurt, min=0.0)
                    kurt_loss = deficit ** 2
                    # Normalize by target_kurt² so loss is O(1) when kurt = 0
                    kurt_loss = kurt_loss / (target_kurt ** 2)
                    loss = loss + lam * kurt_loss
                    aux_total += float(kurt_loss.detach().item()) * lam
                    if stage == "train":
                        self.log("train/aux_kurt_value", kurt, on_epoch=True)
                        self.log("train/aux_kurt_lambda", lam, on_epoch=True)

            # 3. Tail-importance-weighted ε-MSE — re-weight per-sample loss by
            # how extreme the underlying x_0 is (encourages fitting rare windows).
            t_cfg = self.aux_loss_cfg.get("tail", {}) or {}
            if t_cfg.get("enabled", False):
                lam = _ramp(epoch, t_cfg)
                if lam > 0:
                    alpha = float(t_cfg.get("alpha", 0.5))
                    p_pow = float(t_cfg.get("p", 2.0))
                    sigma_data = float(t_cfg.get("sigma_data", 1.0))
                    return_idx = int(t_cfg.get("return_idx", 7))
                    # Window-wise extremity proxy: max |return| over the window
                    extremity = x0_pred[..., return_idx].abs().amax(dim=1)  # (B,)
                    w = 1.0 + alpha * (extremity / sigma_data) ** p_pow      # (B,)
                    tail_loss = (per_sample_mse * w).mean() - per_sample_mse.mean()
                    # Add only the *extra* tail-weighted contribution
                    loss = loss + lam * tail_loss
                    aux_total += float(tail_loss.detach().item()) * lam
                    if stage == "train":
                        self.log("train/aux_tail_loss", tail_loss, on_epoch=True)
                        self.log("train/aux_tail_lambda", lam, on_epoch=True)

            if stage == "train":
                self.log("train/aux_total", aux_total, on_epoch=True)

        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
            if stage == "train":
                self.log("train/cfg_dropout", active_dropout, on_step=False, on_epoch=True)
            # Force-print to stdout so SLURM .out files actually show progress —
            # Lightning's default CSV/TensorBoard logger isn't visible in SLURM logs.
            step = trainer.global_step
            log_every = 20
            if stage == "train" and step % log_every == 0:
                print(
                    f"[{stage}] step {step:>6d}  loss {loss.item():.4f}  "
                    f"pred={self.prediction_type}  p_drop={active_dropout:.2f}",
                    flush=True,
                )
        return loss

    def configure_optimizers(self):
        params = [p for p in self.generator.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
