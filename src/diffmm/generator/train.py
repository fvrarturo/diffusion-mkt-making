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

        if freeze_backbone:
            for p in self.generator.denoiser.parameters():
                p.requires_grad = False

        self.register_buffer("alphas_cumprod", schedule.alphas_cumprod)

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

        # Per-sample MSE (mean over sequence and feature dims) so Min-SNR can
        # weight per-timestep-per-sample.
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
