# MLP init comparison (Xavier vs He vs Uniform)

Quick PyTorch script to compare weight initialization schemes on MNIST with a small MLP.

## Setup
- Recommended: create a virtual environment  
  - Windows (PowerShell): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
- Install deps: `pip install torch torchvision`

## Run a single experiment
Each run logs epoch-wise train/val loss and validation accuracy to stdout.

```
# Xavier + ReLU
python mlp_experiment.py --init xavier --activation relu --epochs 10 --hidden-dim 256 --lr 1e-3

# He (Kaiming) + ReLU
python mlp_experiment.py --init he --activation relu --epochs 10

# Baseline uniform + Tanh
python mlp_experiment.py --init uniform --activation tanh --epochs 10
```

Key flags:
- `--init`: `xavier` | `he` | `uniform`
- `--activation`: `relu` | `tanh`
- `--seed`: set to run multiple trials
- `--hidden-dim`, `--lr`, `--batch-size`, `--epochs`: usual training knobs

## What to do next
- Run 3–5 seeds per init/activation pair, e.g.  
  `for $i in 1 2 3; do python mlp_experiment.py --init xavier --activation relu --seed $i; done`
- Collect logs into a CSV (redirect stdout) and plot learning curves (loss/accuracy vs epochs) to see convergence speed and stability.
- Inspect activation/gradient health: temporarily add hooks to log layer activation mean/std at the first epoch to observe saturation or exploding variance for bad init choices.
- Extend to another activation (e.g., LeakyReLU) and dataset (e.g., Fashion-MNIST) while keeping architecture fixed to test how initialization interacts with nonlinearity and data.
