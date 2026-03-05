from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_dim: int = 512
    learning_rate: float = 1e-4
    batch_size: int = 128
    epochs: int = 15
    hidden_multiplier: int = 2


@dataclass(frozen=True)
class ProbeConfig:
    train_fraction: float = 0.8
    regularization_grid: Sequence[float] = (1e-3, 1e-1, 1.0, 10.0, 1e3)
    random_seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    precision: str = "bf16"
    num_layers: int = 32
    prompt_count: int = 500
    stochastic_rollouts: int = 32
    deterministic_temperature: float = 0.0
    stochastic_temperature: float = 0.7
    hk_delays: Sequence[int] = (0, 2, 4)
    hl_delays: Sequence[int] = (0, 3, 8)
    hl_start_layer: int = 10
    topk_logits_k: int = 256
    fourier_low_freq_choices: Sequence[int] = (10, 32)
    pca_dim: int = 512
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def validate(self) -> None:
        if self.num_layers != 32:
            raise ValueError("Proposal requires all 32 layers.")
        if self.prompt_count != 500:
            raise ValueError("Implementation plan requires 500 paired prompts.")
        if self.stochastic_rollouts != 32:
            raise ValueError("Stochastic setup requires exactly 32 rollouts.")

