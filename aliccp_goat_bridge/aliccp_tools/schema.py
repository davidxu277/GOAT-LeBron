"""Single programmatic entry point for the human-readable AliCCP schema."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).with_name("schema") / "aliccp_schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def feature_ids() -> tuple[str, ...]:
    return tuple(load_schema()["features"].keys())


def feature(field_id: str) -> dict[str, Any]:
    try:
        return load_schema()["features"][str(field_id)]
    except KeyError as error:
        raise KeyError(f"unknown AliCCP field id: {field_id}") from error

