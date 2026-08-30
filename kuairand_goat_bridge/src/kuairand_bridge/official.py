"""Load the vendored official modules without modifying their source."""

from __future__ import annotations

import importlib
import pathlib
import sys


OFFICIAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "official_starter_kit"


def module(name: str):
    path = str(OFFICIAL_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name)
