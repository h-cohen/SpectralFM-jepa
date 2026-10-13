"""Pretrain from a YAML config with optional key=value overrides."""
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
