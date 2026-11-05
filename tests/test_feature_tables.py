import numpy as np
import pandas as pd
from src.data.make_epochs import build_epoch_index
from src.features.build_features import (
    SOMNIA_FEATURE_COLUMNS,
    build_and_write_feature_tables,
    build_basic_feature_table,
    build_rich_feature_table,
    normalize_per_participant,
    somnia_epoch_features,
    write_split_feature_tables,
)


def _write_participant(path, labels, offset):
    n_rows = len(labels)
    values = [offset + index for index in range(n_rows)]
    pd.DataFrame(
        {
            "TIMESTAMP": list(range(n_rows)),
            "BVP": values,
            "ACC_X": values,
            "ACC_Y": values,
            "ACC_Z": values,
            "TEMP": [30.0 + offset] * n_rows,
            "EDA": [0.1 + offset] * n_rows,
            "IBI": [0.8] * n_rows,
            "HR": [70.0] * n_rows,
            "Sleep_Stage": labels,
        }
    ).to_csv(path, index=False)


def test_build_and_write_basic_feature_tables_keep_split_rows_separate(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "W", "P", "P"], 0)
    _write_participant(raw_dir / "S002_whole_df.csv", ["N2", "N2"], 10)
    _write_participant(raw_dir / "S003_whole_df.csv", ["REM", "REM"], 20)
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002", "S003"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(split_path, index=False)
    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)
    epoch_index_path = tmp_path / "epoch_index.csv"
    epoch_index.to_csv(epoch_index_path, index=False)

    features = build_basic_feature_table(raw_dir, epoch_index_path)
    written = write_split_feature_tables(features, tmp_path / "processed")

    assert set(features["label"]) == {"Wake", "Light", "REM"}
    assert "BVP_mean" in features.columns
    assert "ACC_MAG_mean" in features.columns
    assert pd.read_csv(written["train"])["split"].tolist() == ["train"]
    assert pd.read_csv(written["validation"])["split"].tolist() == ["validation"]
    assert pd.read_csv(written["test"])["split"].tolist() == ["test"]


def test_build_rich_feature_table_adds_modular_feature_families(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "W", "N2", "N2"], 0)
    _write_participant(raw_dir / "S002_whole_df.csv", ["REM", "REM"], 10)
    _write_participant(raw_dir / "S003_whole_df.csv", ["N2", "N2"], 20)
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002", "S003"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(split_path, index=False)
    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)
    epoch_index_path = tmp_path / "epoch_index.csv"
    epoch_index.to_csv(epoch_index_path, index=False)

    features = build_rich_feature_table(raw_dir, epoch_index_path)

    feature_columns = [
        c
        for c in features.columns
        if c not in ("participant_id", "epoch_id", "split", "label")
    ]
    assert sorted(feature_columns) == sorted(SOMNIA_FEATURE_COLUMNS)
    assert set(features["split"]) == {"train", "validation", "test"}


def test_somnia_epoch_features_match_formulas():
    n = 750
    arrays = {
        "ACC_X": np.full(n, 1.0),
        "ACC_Y": np.zeros(n),
        "ACC_Z": np.zeros(n),
        "HR": np.array([60.0, 62.0, 64.0]),
        "TEMP": np.array([33.0]),
        "IBI": np.array([1.0, 0.9, np.nan, 1.1, 1.0]),
    }

    features = somnia_epoch_features(arrays)

    assert features["ACC_mean"] == 1.0
    assert features["HR_range"] == 4.0
    assert features["TEMP_mean"] == 33.0
    assert np.isclose(features["IBI_rmssd"], 0.1)
    assert np.isclose(features["IBI_mean"], 1.0)


def test_normalize_per_participant_uses_median_and_p95_p5():
    frame = pd.DataFrame(
        {
            "participant_id": ["S001"] * 5,
            "HR_mean": [50.0, 60.0, 70.0, 80.0, 90.0],
            "ACC_mean": [1.0] * 5,
        }
    )

    normalized = normalize_per_participant(frame)

    assert "HR_mean" not in normalized.columns
    assert "ACC_mean" in normalized.columns
    assert np.isclose(normalized["HR_mean_norm"].iloc[2], 0.0)
    assert np.isclose(normalized["HR_mean_norm"].iloc[4], 20.0 / 36.0)


def test_build_and_write_feature_tables_writes_csv_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "W", "N2", "N2"], 0)
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )
    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)
    epoch_index_path = tmp_path / "epoch_index.csv"
    epoch_index.to_csv(epoch_index_path, index=False)

    written = build_and_write_feature_tables(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index_path,
        output_dir=tmp_path / "processed",
        feature_set="rich",
        manifest_path=tmp_path / "processed" / "feature_manifest.csv",
    )

    manifest = pd.read_csv(written["manifest"])
    assert set(written) == {"train", "validation", "test", "manifest"}
    assert (tmp_path / "processed" / "features_train.csv").exists()
    assert set(manifest["feature"]) == set(SOMNIA_FEATURE_COLUMNS)
    assert {"movement", "cardiovascular", "temperature"} == set(
        manifest["signal_group"]
    )
