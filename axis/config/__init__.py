"""Config loader — reads gesture definitions and action bindings.

Defaults to ./axis/config, overridable via --config or AXIS_CONFIG_DIR.
"""

import json
from pathlib import Path

from axis.cursor.config import CursorConfig, load_cursor_config
from axis.gestures.differentiator import DifferentiatorConfig

CONFIG_DIR = Path(__file__).parent


class Config:
    def __init__(self, config_dir: Path | None = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.gestures = self._load("gestures.json")
        self.actions = self._load("actions.json")
        self.cursor = load_cursor_config(self.config_dir / "cursor.json")
        self.differentiator = DifferentiatorConfig.from_dict(
            self._load("differentiator.json")
        )

    def _load(self, name: str) -> dict:
        path = self.config_dir / name
        with open(path) as f:
            return json.load(f)


def load(config_dir: Path | None = None) -> Config:
    return Config(config_dir)
