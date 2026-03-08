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
    regularization_grid: Sequence[float] = (10.0, 100.0, 1000.0, 5000.0, 10000.0)
    random_seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    precision: str = "bf16"
    num_layers: int = 48
    t0: int = 20
    prompt_count: int = 500
    deterministic_temperature: float = 0.0
    stochastic_temperature: float = 0.7
    stochastic_rollouts: int = 32
    hk_delays: Sequence[int] = (0, 2, 4)
    topk_logits_k: int = 256
    fourier_low_freq_choices: Sequence[int] = (10, 32)
    pca_dim: int = 512
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def validate(self, max_pairs: int | None = None) -> None:
        if self.num_layers != 48:
            raise ValueError("Qwen2.5-14B-Instruct has 48 layers.")
        effective = max_pairs if max_pairs is not None else self.prompt_count
        if effective > self.prompt_count:
            raise ValueError(
                f"max_pairs={effective} exceeds prompt_count={self.prompt_count}."
            )

