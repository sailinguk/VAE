"""Thin, reusable conditional or unconditional VAE built with PyTorch."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


ActivationFactory = Callable[[], nn.Module]


def _mlp(
    dimensions: Sequence[int],
    activation: ActivationFactory,
    activate_last: bool,
) -> nn.Sequential:
    """Create Linear/activation pairs for every adjacent dimension."""
    if len(dimensions) < 1 or any(size <= 0 for size in dimensions):
        raise ValueError("dimensions must contain at least one positive integer")
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        layers.append(nn.Linear(in_features, out_features))
        if activate_last or index < len(dimensions) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Map a vector to a diagonal Gaussian posterior.

    ``dimensions=[128]`` applies no hidden transformation and connects the
    input directly to the posterior heads. For example,
    ``dimensions=[794, 512, 256], latent_dim=20`` creates
    ``794 -> 512 -> 256`` and two ``256 -> 20`` posterior heads.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
        latent_dim: int,
        activation: ActivationFactory = nn.ReLU,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.dimensions = tuple(dimensions)
        if not self.dimensions:
            raise ValueError("dimensions must contain the input dimension")
        self.network = _mlp(self.dimensions, activation, activate_last=True)
        self.mean = nn.Linear(self.dimensions[-1], latent_dim)
        self.log_variance = nn.Linear(self.dimensions[-1], latent_dim)

    def forward(self, vector: torch.Tensor) -> Normal:
        if vector.ndim != 2:
            raise ValueError("Encoder expects [batch, features]; preprocess first")
        hidden = self.network(vector)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden)
        return Normal(mean, torch.exp(0.5 * log_variance))


class Decoder(nn.Module):
    """Map a latent/conditional vector to reconstruction parameters."""

    def __init__(
        self,
        dimensions: Sequence[int],
        activation: ActivationFactory = nn.ReLU,
    ) -> None:
        super().__init__()
        self.dimensions = tuple(dimensions)
        self.network = _mlp(self.dimensions, activation, activate_last=False)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.network(vector)


class VAE(nn.Module):
    """Combine arbitrary vector encoder/decoder layouts and optional labels."""

    def __init__(
        self,
        encoder_dimensions: Sequence[int],
        latent_dim: int,
        decoder_dimensions: Sequence[int],
        activation: ActivationFactory = nn.ReLU,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        if condition_dim < 0:
            raise ValueError("condition_dim cannot be negative")
        if decoder_dimensions[0] != latent_dim + condition_dim:
            raise ValueError(
                "decoder input must equal latent_dim + condition_dim"
            )
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.encoder = Encoder(encoder_dimensions, latent_dim, activation)
        self.decoder = Decoder(decoder_dimensions, activation)

    def _condition(
        self,
        condition: torch.Tensor | None,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.condition_dim == 0:
            if condition is not None:
                raise ValueError("This VAE was created without conditioning")
            return None
        if condition is None:
            raise ValueError("This conditional VAE requires a condition")
        condition = condition.to(device=reference.device, dtype=reference.dtype)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0).expand(batch_size, -1)
        if condition.shape != (batch_size, self.condition_dim):
            raise ValueError(
                f"condition must have shape [{batch_size}, {self.condition_dim}]"
            )
        return condition

    def encode(self, vector: torch.Tensor) -> Normal:
        """Encode a vector whose domain-specific preprocessing is complete."""
        return self.encoder(vector)

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._condition(condition, z.shape[0], z)
        decoder_input = (
            torch.cat((z, condition), dim=-1) if condition is not None else z
        )
        return self.decoder(decoder_input)

    def forward(
        self,
        vector: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Normal]:
        posterior = self.encode(vector)
        return self.decode(posterior.rsample(), condition), posterior


def kl_loss(posterior: Normal, prior: Normal | None = None) -> torch.Tensor:
    """Return one KL value per example, leaving reduction to the caller."""
    if prior is None:
        prior = Normal(
            torch.zeros_like(posterior.loc),
            torch.ones_like(posterior.scale),
        )
    return kl_divergence(posterior, prior).flatten(start_dim=1).sum(dim=1)


def vae_loss(
    reconstruction_loss: torch.Tensor,
    posterior: Normal,
    beta: float = 1.0,
    prior: Normal | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine per-example domain reconstruction losses with the KL term."""
    reconstruction = reconstruction_loss.flatten()
    kl = kl_loss(posterior, prior)
    if reconstruction.shape != kl.shape:
        raise ValueError("reconstruction_loss and KL need one value per example")
    total = reconstruction + beta * kl
    if reduction == "none":
        return total, reconstruction, kl
    if reduction == "mean":
        return total.mean(), reconstruction.mean(), kl.mean()
    if reduction == "sum":
        return total.sum(), reconstruction.sum(), kl.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


@torch.no_grad()
def generate(
    model: VAE,
    num_samples: int,
    condition: torch.Tensor | None = None,
    temperature: float = 1.0,
    output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sample the prior and decode with an optional fixed/per-sample condition."""
    if num_samples <= 0 or temperature <= 0:
        raise ValueError("num_samples and temperature must be positive")
    parameter = next(model.parameters())
    prior = Normal(
        torch.zeros(model.latent_dim, device=parameter.device, dtype=parameter.dtype),
        torch.full(
            (model.latent_dim,),
            temperature,
            device=parameter.device,
            dtype=parameter.dtype,
        ),
    )
    output = model.decode(prior.sample((num_samples,)), condition)
    return output_transform(output) if output_transform is not None else output
