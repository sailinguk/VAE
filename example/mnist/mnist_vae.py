"""Unconditional or label-conditional MNIST example for ../../../vae.py."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vae import VAE, generate, vae_loss


ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--condition-dim", type=int, default=10)
    parser.add_argument("--encoder-dimensions", type=int, nargs="+")
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--decoder-dimensions", type=int, nargs="+")
    parser.add_argument("--activation", choices=ACTIVATIONS, default="relu")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--num-generated", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--test-subset", type=int)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    condition_dim = args.condition_dim if args.conditional else 0
    if args.conditional and condition_dim != 10:
        parser.error("MNIST one-hot labels require --condition-dim 10")
    if args.encoder_dimensions is None:
        args.encoder_dimensions = [784 + condition_dim, 400]
    if args.decoder_dimensions is None:
        args.decoder_dimensions = [args.latent_dim + condition_dim, 400, 784]
    return args


def labels_to_condition(labels: torch.Tensor, condition_dim: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=condition_dim).to(torch.float32)


def limited(dataset, count: int | None):
    return dataset if count is None else Subset(dataset, range(min(count, len(dataset))))


def epoch(model, loader, optimizer, reconstruction_function, beta, device):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(3)
    examples = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            vector = images.flatten(start_dim=1).to(device)
            condition = (
                labels_to_condition(labels, model.condition_dim).to(device)
                if model.condition_dim
                else None
            )
            encoder_vector = (
                torch.cat((vector, condition), dim=-1)
                if condition is not None
                else vector
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, posterior = model(encoder_vector, condition)
            reconstruction = reconstruction_function(logits, vector).sum(dim=1)
            losses = vae_loss(reconstruction, posterior, beta=beta)
            if training:
                losses[0].backward()
                optimizer.step()
            values = torch.tensor([value.detach().item() for value in losses])
            totals += values * images.shape[0]
            examples += images.shape[0]
    metrics = tuple((totals / examples).tolist())
    if not all(math.isfinite(value) for value in metrics):
        raise RuntimeError(f"Non-finite metrics detected: {metrics}")
    return metrics


def checkpoint_config(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    run_name = "conditional" if args.conditional else "unconditional"
    output_dir = (
        args.run_dir if args.run_dir is not None else args.output_dir / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_data = limited(
        datasets.MNIST(args.data_dir, train=True, download=args.download, transform=transform),
        args.train_subset,
    )
    test_data = limited(
        datasets.MNIST(args.data_dir, train=False, download=args.download, transform=transform),
        args.test_subset,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    condition_dim = args.condition_dim if args.conditional else 0
    model = VAE(
        args.encoder_dimensions,
        args.latent_dim,
        args.decoder_dimensions,
        ACTIVATIONS[args.activation],
        condition_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    reconstruction_function = nn.BCEWithLogitsLoss(reduction="none")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_reconstruction",
                "train_kl",
                "test_total",
                "test_reconstruction",
                "test_kl",
            ]
        )
        for number in range(1, args.epochs + 1):
            train = epoch(model, train_loader, optimizer, reconstruction_function, args.beta, device)
            test = epoch(model, test_loader, None, reconstruction_function, args.beta, device)
            writer.writerow([number, *train, *test])
            handle.flush()
            print(
                f"mode={run_name} epoch={number:03d} train={train[0]:.4f} "
                f"test={test[0]:.4f} reconstruction={test[1]:.4f} kl={test[2]:.4f}",
                flush=True,
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "encoder_dimensions": list(args.encoder_dimensions),
        "decoder_dimensions": list(args.decoder_dimensions),
        "latent_dim": args.latent_dim,
        "condition_dim": condition_dim,
        "config": checkpoint_config(args),
    }
    torch.save(checkpoint, output_dir / "vae.pt")

    if args.conditional:
        labels = torch.arange(10).repeat_interleave(args.samples_per_class)
        condition = labels_to_condition(labels, condition_dim).to(device)
        count = len(labels)
        nrow = args.samples_per_class
    else:
        condition = None
        count = args.num_generated
        nrow = 8
    samples = generate(model, count, condition, args.temperature, torch.sigmoid)
    save_image(samples.view(-1, 1, 28, 28), output_dir / "generated.png", nrow=nrow)


if __name__ == "__main__":
    main()
