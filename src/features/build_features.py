"""CSV feature-table construction for DREAMT epoch summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_SIGNAL_COLUMNS, SPLIT_ORDER, TARGET_LABELS
from src.data.harmonize import HARMONIZED_SIGNAL_COLUMNS, harmonize_dreamt_epoch
from src.data.load_dreamt import extract_participant_id, list_participant_csvs
from src.features.acc_features import ACC_MAGNITUDE_COLUMN, accelerometer_magnitude
from src.features.base_features import summarize_epoch, summarize_signal
from src.features.feature_manifest import write_feature_manifest

FEATURE_ID_COLUMNS = ["participant_id", "epoch_id", "split", "label"]
SPLIT_OUTPUT_NAMES = {"train": "train", "validation": "val", "test": "test"}
FEATURE_SETS = ("basic", "rich")
NORMALIZED_PREFIXES = ("HR_", "IBI_", "TEMP_")
NORMALIZED_SUFFIX = "_norm"
SOMNIA_FEATURE_COLUMNS = (
    "ACC_mean",
    "ACC_std",
    "ACC_iqr",
    "ACC_max",
    "HR_mean_norm",
    "HR_std_norm",
    "HR_min_norm",
    "HR_max_norm",
    "HR_range_norm",
    "IBI_mean_norm",
    "IBI_sdnn_norm",
    "IBI_rmssd_norm",
    "TEMP_mean_norm",
)
EPOCH_INDEX_STRING_COLUMNS = {
    "participant_id": str,
    "split": str,
    "raw_label": "string",
    "mapped_label": "string",
    "exclusion_reason": "string",
    "segmentation_reason": "string",
}


def build_basic_feature_table(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
) -> pd.DataFrame:
    """Build one basic statistical feature row per valid 30-second epoch.

    Raw DREAMT files are intentionally loaded one participant at a time. The full
    64 Hz dataset is large enough that keeping every participant CSV in memory at
    once can exceed RAM on a normal workstation.
    """
    epoch_index = load_valid_epoch_index(epoch_index_path)
    participant_paths = _participant_path_lookup(raw_dir)
    rows: list[dict[str, object]] = []

    for participant_id, participant_epochs in epoch_index.groupby(
        "participant_id", sort=True
    ):
        participant_id = str(participant_id)
        if participant_id not in participant_paths:
            raise ValueError(f"Raw CSV for {participant_id} is missing from {raw_dir}.")

        participant = _load_participant_signal_frame(participant_paths[participant_id])
        for epoch_row in participant_epochs.itertuples(index=False):
            epoch = participant.iloc[int(epoch_row.start_row) : int(epoch_row.end_row)]
            features = _basic_epoch_features(epoch)
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": int(epoch_row.epoch_id),
                    "split": str(epoch_row.split),
                    "label": str(epoch_row.mapped_label),
                    **features,
                }
            )

    if not rows:
        return pd.DataFrame(columns=FEATURE_ID_COLUMNS)

    features = pd.DataFrame(rows)
    feature_columns = sorted(
        column for column in features.columns if column not in FEATURE_ID_COLUMNS
    )
    return features[[*FEATURE_ID_COLUMNS, *feature_columns]]


def build_rich_feature_table(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
) -> pd.DataFrame:
    """Build the Somnia feature table from harmonized 30-second epochs.

    Each epoch is harmonized first, then movement, heart-rate, IBI, and
    temperature features are computed. Heart-rate, IBI, and temperature
    features are normalized per participant with the median and the
    5th-95th percentile range.
    """
    epoch_index = load_valid_epoch_index(epoch_index_path)
    participant_paths = _participant_path_lookup(raw_dir)
    rows: list[dict[str, object]] = []

    for participant_id, participant_epochs in epoch_index.groupby(
        "participant_id", sort=True
    ):
        participant_id = str(participant_id)
        if participant_id not in participant_paths:
            raise ValueError(f"Raw CSV for {participant_id} is missing from {raw_dir}.")

        participant = _load_participant_signal_frame(
            participant_paths[participant_id], HARMONIZED_SIGNAL_COLUMNS
        )
        signal_arrays = {
            column: pd.to_numeric(participant[column], errors="coerce").to_numpy(
                dtype=float
            )
            for column in HARMONIZED_SIGNAL_COLUMNS
            if column in participant.columns
        }
        for epoch_row in participant_epochs.itertuples(index=False):
            epoch_arrays = harmonize_dreamt_epoch(
                signal_arrays, int(epoch_row.start_row), int(epoch_row.end_row)
            )
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": int(epoch_row.epoch_id),
                    "split": str(epoch_row.split),
                    "label": str(epoch_row.mapped_label),
                    **somnia_epoch_features(epoch_arrays),
                }
            )

    if not rows:
        return pd.DataFrame(columns=FEATURE_ID_COLUMNS)
    features = normalize_per_participant(pd.DataFrame(rows))
    return _ordered_feature_table(features)


def somnia_epoch_features(epoch_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute the Somnia features for one harmonized epoch."""
    features: dict[str, float] = {}

    magnitude = _acc_magnitude(epoch_arrays)
    features["ACC_mean"] = _mean(magnitude)
    features["ACC_std"] = _std(magnitude)
    features["ACC_iqr"] = _iqr(magnitude)
    features["ACC_max"] = _max(magnitude)

    hr = _finite(epoch_arrays.get("HR"))
    features["HR_mean"] = _mean(hr)
    features["HR_std"] = _std(hr)
    features["HR_min"] = _min(hr)
    features["HR_max"] = _max(hr)
    features["HR_range"] = features["HR_max"] - features["HR_min"]

    ibi = np.asarray(epoch_arrays.get("IBI", []), dtype=float)
    beats = _finite(ibi)
    diffs = np.diff(ibi)
    diffs = diffs[np.isfinite(diffs)]
    features["IBI_mean"] = _mean(beats)
    features["IBI_sdnn"] = _std(beats)
    features["IBI_rmssd"] = (
        float(np.sqrt(np.mean(np.square(diffs)))) if diffs.size else np.nan
    )

    features["TEMP_mean"] = _mean(_finite(epoch_arrays.get("TEMP")))
    return features


def normalize_per_participant(features: pd.DataFrame) -> pd.DataFrame:
    """Replace HR, IBI, and TEMP features with (x - median) / (P95 - P5)."""
    features = features.copy()
    columns = [
        column for column in features.columns if column.startswith(NORMALIZED_PREFIXES)
    ]
    for column in columns:
        grouped = features.groupby("participant_id")[column]
        median = grouped.transform("median")
        spread = grouped.transform(lambda values: values.quantile(0.95)) - (
            grouped.transform(lambda values: values.quantile(0.05))
        )
        spread = spread.where(spread > 0)
        features[f"{column}{NORMALIZED_SUFFIX}"] = (features[column] - median) / spread
    return features.drop(columns=columns)


def write_split_feature_tables(
    features: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Write train/validation/test feature CSVs using prior-project filenames."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        split_rows = features[features["split"] == split].copy()
        filename = f"features_{SPLIT_OUTPUT_NAMES[split]}.csv"
        output_path = output_root / filename
        split_rows.to_csv(output_path, index=False)
        written[split] = output_path
    return written


def build_and_write_basic_feature_tables(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Build basic features and write split CSV files."""
    features = build_basic_feature_table(raw_dir, epoch_index_path)
    return write_split_feature_tables(features, output_dir)


def build_and_write_rich_feature_tables(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    output_dir: str | Path = "data/processed",
    manifest_path: str | Path = "data/processed/feature_manifest.csv",
) -> dict[str, Path]:
    """Build rich features, write split CSV files, and write a CSV manifest."""
    features = build_rich_feature_table(raw_dir, epoch_index_path)
    written = write_split_feature_tables(features, output_dir)
    written["manifest"] = write_feature_manifest(features, manifest_path)
    return written


def build_and_write_feature_tables(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    output_dir: str | Path = "data/processed",
    *,
    feature_set: str = "rich",
    manifest_path: str | Path = "data/processed/feature_manifest.csv",
) -> dict[str, Path]:
    """Build and write the requested feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS}; got {feature_set}"
        )
    if feature_set == "basic":
        features = build_basic_feature_table(raw_dir, epoch_index_path)
        written = write_split_feature_tables(features, output_dir)
        written["manifest"] = write_feature_manifest(features, manifest_path)
        return written
    return build_and_write_rich_feature_tables(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )


def load_valid_epoch_index(epoch_index_path: str | Path) -> pd.DataFrame:
    """Load valid target-labeled epochs from the epoch-index CSV."""
    path = Path(epoch_index_path)
    if not path.exists():
        raise FileNotFoundError(f"Epoch index CSV does not exist: {path}")

    epoch_index = pd.read_csv(path, dtype=EPOCH_INDEX_STRING_COLUMNS)
    required = {
        "participant_id",
        "split",
        "epoch_id",
        "start_row",
        "end_row",
        "mapped_label",
        "is_valid_epoch",
    }
    missing = sorted(required - set(epoch_index.columns))
    if missing:
        raise ValueError(f"epoch index is missing column(s): {missing}")

    valid = epoch_index[epoch_index["is_valid_epoch"].astype(bool)].copy()
    valid = valid[valid["mapped_label"].isin(TARGET_LABELS)].copy()
    return valid.sort_values(["participant_id", "epoch_id"]).reset_index(drop=True)


def _participant_path_lookup(raw_dir: str | Path) -> dict[str, Path]:
    """Return participant CSV paths keyed by participant ID."""
    return {
        extract_participant_id(path): path for path in list_participant_csvs(raw_dir)
    }


def _load_participant_signal_frame(
    path: str | Path, columns: tuple[str, ...] = RAW_SIGNAL_COLUMNS
) -> pd.DataFrame:
    """Load only signal columns needed for feature extraction."""
    csv_path = Path(path)
    available_columns = pd.read_csv(csv_path, nrows=0).columns
    usecols = [column for column in columns if column in available_columns]
    if not usecols:
        raise ValueError(f"{csv_path} does not contain any expected signal columns.")
    return pd.read_csv(csv_path, usecols=usecols)


def _basic_epoch_features(epoch: pd.DataFrame) -> dict[str, float]:
    """Compute first-milestone statistical features for one epoch."""
    features = summarize_epoch(epoch, RAW_SIGNAL_COLUMNS)
    if all(column in epoch.columns for column in ("ACC_X", "ACC_Y", "ACC_Z")):
        acc_mag = accelerometer_magnitude(epoch)
        features.update(summarize_signal(acc_mag, ACC_MAGNITUDE_COLUMN))
    return features


def _ordered_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = sorted(
        column for column in features.columns if column not in FEATURE_ID_COLUMNS
    )
    return features[[*FEATURE_ID_COLUMNS, *feature_columns]]


def _finite(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _acc_magnitude(epoch_arrays: dict[str, np.ndarray]) -> np.ndarray:
    if not all(column in epoch_arrays for column in ("ACC_X", "ACC_Y", "ACC_Z")):
        return np.asarray([], dtype=float)
    axes = np.vstack([epoch_arrays[column] for column in ("ACC_X", "ACC_Y", "ACC_Z")])
    valid = np.isfinite(axes).all(axis=0)
    return np.sqrt(np.square(axes[:, valid]).sum(axis=0))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def _iqr(values: np.ndarray) -> float:
    if not values.size:
        return np.nan
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float(q75 - q25)


def _min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else np.nan


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else np.nan
