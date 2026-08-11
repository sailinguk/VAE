# VAE

A reusable PyTorch VAE with an MNIST example.

## Requirements

- Python 3.10+
- PyTorch
- torchvision (for the MNIST example)

Install versions suitable for your own operating system, CUDA version, and
environment.

## MNIST example parameters

- `--conditional`: enable conditional generation with digit labels
- `--condition-dim`: condition-vector size (10 for MNIST)
- `--encoder-dimensions`: encoder input size and hidden-layer widths
- `--latent-dim`: latent-space dimension
- `--decoder-dimensions`: decoder input size, hidden-layer widths, and output size
- `--activation`: `relu`, `leaky_relu`, or `tanh`
- `--beta`: KL-loss weight
- `--epochs`: number of training epochs
- `--batch-size`: batch size
- `--learning-rate`: Adam learning rate
- `--weight-decay`: Adam weight decay
- `--num-workers`: number of DataLoader workers
- `--seed`: random seed
- `--device`: `cpu` or `cuda`
- `--data-dir`: MNIST data directory
- `--output-dir`: parent output directory; the mode name is added automatically
- `--run-dir`: exact output directory, overriding `--output-dir`
- `--num-generated`: number of unconditional samples to generate
- `--samples-per-class`: conditional samples generated per digit
- `--temperature`: latent sampling temperature
- `--train-subset`, `--test-subset`: optional dataset-size limits
- `--download`, `--no-download`: enable or disable MNIST download

Show defaults and all available arguments with:

```bash
python3 example/mnist/mnist_vae.py --help
```

## Example

Run an unconditional VAE with `784 -> 400 -> z=20` in the encoder and
`z=20 -> 400 -> 784` in the decoder:

```bash
python3 example/mnist/mnist_vae.py \
  --device cuda \
  --epochs 100 \
  --latent-dim 20 \
  --encoder-dimensions 784 400 \
  --decoder-dimensions 20 400 784
```

If you run this on an HPC cluster, you may need to place the command in a Slurm sbatch script or another scheduler-specific job submission script.
