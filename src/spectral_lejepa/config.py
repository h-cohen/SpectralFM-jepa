"""Experiment configuration: one YAML file plus `key.sub=value` command-line overrides.

The resolved dictionary is the single source of truth for a run: it is saved next to
the checkpoints and logged to W&B in full.
"""
from __future__ import annotations

from typing import Any, Sequence

import yaml


def parse_value(raw: str) -> Any:
    """YAML-parse an override value. PyYAML reads `1e-3` as a string, so retry as float."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path, overrides: Sequence[str] = ()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        *sections, leaf = key.split(".")
        node = cfg
        for section in sections:
            if not isinstance(node.get(section), dict):
                raise KeyError(f"unknown config section {section!r} in override {item!r}")
            node = node[section]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r} (typo?)")
        node[leaf] = parse_value(raw)
    return cfg
