import pandas as pd
from src.data.make_epochs import (
    audit_epoch_alignment,
    build_epoch_index,
    infer_label_aligned_segments,
)


def _write_participant(path, labels):
    n_rows = len(labels)
    pd.DataFrame(
        {
            "TIMESTAMP": list(range(n_rows)),
            "BVP": range(n_rows),
            "ACC_X": range(n_rows),
            "ACC_Y": range(n_rows),
            "ACC_Z": range(n_rows),
            "TEMP": [30.0] * n_rows,
            "EDA": [0.1] * n_rows,
            "IBI": [0.8] * n_rows,
            "HR": [70.0] * n_rows,
            "Sleep_Stage": labels,
        }
    ).to_csv(path, index=False)


def test_build_epoch_index_maps_labels_and_excludes_preparation(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(
        raw_dir / "S001_whole_df.csv", ["P", "P", "W", "W", "N2", "N2", "REM", "REM"]
    )
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)

    assert len(epoch_index) == 4
    assert not bool(epoch_index.loc[0, "is_valid_epoch"])
    assert epoch_index.loc[0, "exclusion_reason"] == "excluded_label"
    assert epoch_index.loc[1:, "mapped_label"].tolist() == ["Wake", "Light", "REM"]
    assert set(epoch_index["split"]) == {"train"}


def test_build_epoch_index_rejects_participants_missing_from_split(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "W"])
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S002"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    try:
        build_epoch_index(raw_dir, split_path, rows_per_epoch=2)
    except ValueError as exc:
        assert "missing from split assignments" in str(exc)
    else:
        raise AssertionError("Expected missing split assignment to raise ValueError")


def test_build_epoch_index_can_infer_nonzero_start_offset(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(
        raw_dir / "S001_whole_df.csv",
        ["P", "W", "W", "N2", "N2", "REM", "REM"],
    )
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)

    assert epoch_index["epoch_start_offset_rows"].unique().tolist() == [1]
    assert epoch_index["mapped_label"].tolist() == ["Wake", "Light", "REM"]


def test_build_epoch_index_marks_mixed_label_epochs_invalid(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "N2"])
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    epoch_index = build_epoch_index(
        raw_dir, split_path, rows_per_epoch=2, infer_start_offset=False
    )

    assert not bool(epoch_index.loc[0, "is_valid_epoch"])
    assert epoch_index.loc[0, "exclusion_reason"] == "label_changed_within_epoch"


def test_audit_epoch_alignment_reports_multi_phase_transition_remainders():
    labels = pd.Series(
        [
            "P",
            "P",
            "P",
            "P",
            "P",
            "W",
            "W",
            "W",
            "W",
            "N2",
            "N2",
            "N2",
            "N2",
            "W",
            "W",
            "W",
            "W",
            "Missing",
            "Missing",
            "Missing",
            "Missing",
            "Missing",
            "Missing",
            "N2",
            "N2",
            "N2",
            "N2",
            "W",
            "W",
            "W",
            "W",
            "REM",
            "REM",
            "REM",
            "REM",
        ]
    )

    audit = audit_epoch_alignment(
        labels,
        expected_n_rows=4,
        min_transition_agreement=0.80,
        candidate_offset_step=1,
    )

    assert audit["top_remainder"] == 1
    assert audit["top_transition_count"] == 4
    assert audit["second_remainder"] == 3
    assert audit["second_transition_count"] == 3
    assert audit["inferred_offset"] == 0
    assert audit["best_offset"] == 1
    assert audit["current_mapped_mixed_epochs"] > audit["best_mapped_mixed_epochs"]


def test_segmented_epoching_splits_around_long_unmapped_blocks(tmp_path):
    labels = [
        "P",
        "P",
        "P",
        "P",
        "P",
        "W",
        "W",
        "W",
        "W",
        "N2",
        "N2",
        "N2",
        "N2",
        "W",
        "W",
        "W",
        "W",
        "Missing",
        "Missing",
        "Missing",
        "Missing",
        "Missing",
        "Missing",
        "N2",
        "N2",
        "N2",
        "N2",
        "W",
        "W",
        "W",
        "W",
        "REM",
        "REM",
        "REM",
        "REM",
    ]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", labels)
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    legacy = build_epoch_index(raw_dir, split_path, rows_per_epoch=4)
    segmented = build_epoch_index(
        raw_dir,
        split_path,
        rows_per_epoch=4,
        segmented_epoching=True,
        min_segment_boundary_unmapped_epochs=1,
    )

    assert legacy["raw_label"].astype(str).str.contains(r"\|").any()
    assert segmented["raw_label"].tolist() == ["W", "N2", "W", "N2", "W", "REM"]
    assert segmented["mapped_label"].tolist() == [
        "Wake",
        "Light",
        "Wake",
        "Light",
        "Wake",
        "REM",
    ]
    assert segmented["epoch_start_offset_rows"].tolist() == [1, 1, 1, 3, 3, 3]
    assert segmented["segment_id"].tolist() == [0, 0, 0, 1, 1, 1]
    assert segmented["is_valid_epoch"].astype(bool).all()


def test_infer_label_aligned_segments_uses_long_unmapped_boundaries_only():
    labels = pd.Series(["P", "P", "P", "P", "W", "W", "N2", "N2", "W", "W", "N2", "N2"])

    segments = infer_label_aligned_segments(
        labels,
        expected_n_rows=2,
        min_segment_boundary_unmapped_rows=3,
    )

    assert len(segments) == 1
    assert segments[0].start_row == 4
    assert segments[0].end_row == 12
    assert segments[0].offset_rows == 0
