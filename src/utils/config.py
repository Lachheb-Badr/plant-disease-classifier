"""Configuration loading."""
from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Load the YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
