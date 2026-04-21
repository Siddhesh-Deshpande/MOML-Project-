from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    data: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def output_dir(self) -> Path:
        return Path(self.data["output_dir"])


def load_config(config_path: str | Path) -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return Config(data=payload)
