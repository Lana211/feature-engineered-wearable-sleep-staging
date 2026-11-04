import pandas as pd
from src.data.make_epochs import build_epoch_index
from src.features.build_features import (
    _participant_zscore_source_columns,
    build_and_write_feature_tables,
    build_basic_feature_table,
    build_rich_feature_table,
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

    features = build_rich_feature_table(
        raw_dir,
        epoch_index_path,
        rolling_windows=(3,),
        context_source_features=("BVP_mean", "ACC_MAG_mean", "IBI_rmssd"),
    )

    assert "BVP_mean" in features.columns
    assert "ACC_motion_intensity" in features.columns
    assert "IBI_rmssd" in features.columns
    assert "IBI_sdnn" not in features.columns
    assert "BVP_mean_roll3_mean" in features.columns
    assert "BVP_mean_roll3_std" in features.columns
    assert "BVP_mean_participant_z" in features.columns
    assert "BVP_valid_fraction" in features.columns
    assert "BVP_valid_fraction_participant_z" not in features.columns
    assert "ACC_MAG_valid_fraction_participant_z" not in features.columns
    assert "_segment_id" not in features.columns
    assert set(features["split"]) == {"train", "validation", "test"}


def test_participant_zscore_sources_skip_valid_fraction_features():
    source_columns = _participant_zscore_source_columns(
        [
            "BVP_mean",
            "BVP_valid_fraction",
            "ACC_MAG_valid_fraction",
            "HR_abs_diff_mean",
        ]
    )

    assert source_columns == ["BVP_mean", "HR_abs_diff_mean"]


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
    assert {
        "basic_statistical",
        "signal_specific",
        "rolling_context",
        "whole_night_subject_normalized",
    }.issubset(set(manifest["feature_family"]))
    assert "cardiovascular" in set(manifest["signal_group"])
