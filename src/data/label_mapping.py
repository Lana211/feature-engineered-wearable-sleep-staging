"""DREAMT PSG label mapping for the three-class target."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.config import TARGET_LABELS

PREPARATION_LABELS = {"P", "PREP", "PREPARATION"}
WAKE_LABELS = {"W", "WAKE"}
LIGHT_LABELS = {"N1", "N2"}
DEEP_LABELS = {"N3"}
REM_LABELS = {"R", "REM"}


def normalize_stage_text(stage: object) -> str | None:
    """Normalize raw label text while preserving missing values."""
    if pd.isna(stage):
        return None
    text = str(stage).strip().upper()
    return text or None


def map_sleep_stage(stage: object) -> str | None:
    """Map DREAMT sleep-stage labels to Wake, Light, Deep, REM, or exclusion."""    normalized = normalize_stage_text(stage)
    if normalized is None or normalized in PREPARATION_LABELS:
        return None
    if normalized in WAKE_LABELS:
        return "Wake"
    if normalized in LIGHT_LABELS:
        return "Light"
    if normalized in DEEP_LABELS:
        return "Deep"
    if normalized in REM_LABELS:
        return "REM"
    return None


def map_sleep_stage_series(stages: Iterable[object]) -> pd.Series:
    """Vectorized label mapping that returns pandas nullable strings."""
    mapped = pd.Series([map_sleep_stage(stage) for stage in stages], dtype="string")
    return mapped


def is_target_label(label: object) -> bool:
    """Return true when a mapped label is one of the modeled classes."""
    return label in TARGET_LABELS
