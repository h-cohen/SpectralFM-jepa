"""Pretrain LeJEPA on SpectralFM spectra.

  uv run python scripts/pretrain.py [--config configs/pretrain.yaml] [key.sub=value ...]
  e.g. uv run python scripts/pretrain.py training.max_steps=300 data.max_train_samples=20000
"""
import argparse

from spectral_lejepa.config import load_config
from spectral_lejepa.training.trainer import train


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    a = ap.parse_args()
    result = train(load_config(a.config, a.overrides))
    print("output dir:", result["output_dir"])
    print("final checkpoint:", result["final_checkpoint"])


if __name__ == "__main__":
    main()
