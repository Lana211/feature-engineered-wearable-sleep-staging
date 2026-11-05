import numpy as np
from src.data.harmonize import (
    block_mean,
    harmonize_dreamt_epoch,
    real_ibi_beats,
    resample_linear,
)


def test_resample_linear_converts_30_seconds_from_64_to_25_hz():
    values = np.arange(30 * 64, dtype=float)

    resampled = resample_linear(values, 64, 25)

    assert resampled.size == 750
    assert resampled[0] == 0.0


def test_block_mean_gives_one_hr_value_per_second():
    values = np.r_[np.full(64, 60.0), np.full(64, 70.0)]

    assert block_mean(values, 64).tolist() == [60.0, 70.0]


def test_real_ibi_beats_drops_gaps_and_marks_break():
    rate = 64
    beats = [1.0] * 6 + [1.0] * 6
    runs = [
        np.full(int(value * rate), value + index * 0.01)
        for index, value in enumerate(beats)
    ]
    gap = np.full(10 * rate, 0.9)
    values = np.concatenate([*runs[:6], gap, *runs[6:]])

    result = real_ibi_beats(values, rate, min_beats=10)

    assert np.isfinite(result).sum() == 12
    assert np.isnan(result[6])


def test_real_ibi_beats_returns_nan_when_too_few_beats():
    values = np.full(30 * 64, 0.9)

    result = real_ibi_beats(values, 64)

    assert np.isnan(result).all()


def test_harmonize_dreamt_epoch_outputs_shared_format():
    n = 30 * 64
    arrays = {
        "ACC_X": np.full(n, 64.0),
        "ACC_Y": np.zeros(n),
        "ACC_Z": np.zeros(n),
        "HR": np.full(n, 60.0),
        "TEMP": np.full(n, 33.0),
        "IBI": np.full(n, 1.0),
        "BVP": np.zeros(n),
        "EDA": np.zeros(n),
    }

    result = harmonize_dreamt_epoch(arrays, 0, n)

    assert set(result) == {"ACC_X", "ACC_Y", "ACC_Z", "HR", "TEMP", "IBI"}
    assert result["ACC_X"].size == 750
    assert np.allclose(result["ACC_X"], 1.0)
    assert result["HR"].size == 30
    assert result["TEMP"].tolist() == [33.0]
