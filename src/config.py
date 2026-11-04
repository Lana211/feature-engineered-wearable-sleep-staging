"""Project-wide constants and path defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_TITLE = (
    "Feature-Engineered Machine Learning for Wearable Sleep Stage Classification"
)
TARGET_LABELS = ("Wake", "Light", "Deep", "REM")
SPLIT_ORDER = ("train", "validation", "test")

RAW_SIGNAL_COLUMNS = (
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "TEMP",
    "EDA",
    "IBI",
    "HR",
)
ACC_AXIS_COLUMNS = ("ACC_X", "ACC_Y", "ACC_Z")

PARTICIPANT_PATTERN = "S*_whole_df.csv"
LABEL_COLUMN = "Sleep_Stage"
TIME_COLUMN = "TIMESTAMP"
EPOCH_SECONDS = 30

PREVIOUS_PROJECT_ROOT = Path("/home/manns79/dreamt-wearable-sleep-staging")
PREVIOUS_PROJECT_SPLIT_PATH = (
    PREVIOUS_PROJECT_ROOT / "data" / "interim" / "split_assignments.csv"
)


@dataclass(frozen=True)
class ProjectPaths:
    """Default project paths for local scripts."""

    root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    split_assignments: Path = Path("data/interim/split_assignments.csv")
    epoch_index: Path = Path("data/interim/epoch_index.csv")
    features_train: Path = Path("data/processed/features_train.csv")
    features_val: Path = Path("data/processed/features_val.csv")
    features_test: Path = Path("data/processed/features_test.csv")
    feature_manifest: Path = Path("data/processed/feature_manifest.csv")
