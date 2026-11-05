"""Convert device epochs to one shared signal format before feature extraction."""

from __future__ import annotations

import numpy as np

HARMONIZED_SIGNAL_COLUMNS = ("ACC_X", "ACC_Y", "ACC_Z", "TEMP", "IBI", "HR")
ACC_AXIS_COLUMNS = ("ACC_X", "ACC_Y", "ACC_Z")

DREAMT_SAMPLING_RATE_HZ = 64
DREAMT_ACC_UNITS_PER_G = 64.0

ACC_TARGET_RATE_HZ = 25
HR_TARGET_RATE_HZ = 1

IBI_GAP_FACTOR = 1.5
MIN_REAL_BEATS = 10


def harmonize_dreamt_epoch(
    signal_arrays: dict[str, np.ndarray],
    start_row: int,
    end_row: int,
    sampling_rate_hz: int = DREAMT_SAMPLING_RATE_HZ,
) -> dict[str, np.ndarray]:
    """Return one DREAMT epoch as ACC 25 Hz in g, HR 1 Hz, real IBI beats, TEMP mean."""
    harmonized: dict[str, np.ndarray] = {}

    if all(column in signal_arrays for column in ACC_AXIS_COLUMNS):
        for column in ACC_AXIS_COLUMNS:
            acc_g = signal_arrays[column][start_row:end_row] / DREAMT_ACC_UNITS_PER_G
            harmonized[column] = resample_linear(
                acc_g, sampling_rate_hz, ACC_TARGET_RATE_HZ
            )

    if "HR" in signal_arrays:
        harmonized["HR"] = block_mean(
            signal_arrays["HR"][start_row:end_row],
            sampling_rate_hz // HR_TARGET_RATE_HZ,
        )

    if "TEMP" in signal_arrays:
        harmonized["TEMP"] = epoch_mean(signal_arrays["TEMP"][start_row:end_row])

    if "IBI" in signal_arrays:
        harmonized["IBI"] = real_ibi_beats(
            signal_arrays["IBI"][start_row:end_row], sampling_rate_hz
        )

    return harmonized


def resample_linear(
    values: np.ndarray, source_rate_hz: float, target_rate_hz: float
) -> np.ndarray:
    """Resample a uniformly sampled signal to a new rate by linear interpolation."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    duration = values.size / source_rate_hz
    target_size = max(1, int(round(duration * target_rate_hz)))
    source_times = np.arange(values.size) / source_rate_hz
    target_times = np.arange(target_size) / target_rate_hz
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full(target_size, np.nan)
    return np.interp(target_times, source_times[valid], values[valid])


def block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    """Average consecutive blocks of samples, keeping a final partial block."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    means = []
    for start in range(0, values.size, block_size):
        block = values[start : start + block_size]
        block = block[np.isfinite(block)]
        means.append(float(np.mean(block)) if block.size else np.nan)
    return np.asarray(means, dtype=float)


def epoch_mean(values: np.ndarray) -> np.ndarray:
    """Return the epoch mean as a one-value array."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return np.asarray([float(np.mean(valid)) if valid.size else np.nan])


def real_ibi_beats(
    values: np.ndarray,
    sampling_rate_hz: float,
    gap_factor: float = IBI_GAP_FACTOR,
    min_beats: int = MIN_REAL_BEATS,
) -> np.ndarray:
    """Return one value per real beat, with NaN where a gap breaks the sequence."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(1, np.nan)

    change = np.r_[True, values[1:] != values[:-1]]
    run_start = np.flatnonzero(change)
    run_end = np.r_[run_start[1:], values.size]
    run_value = values[run_start]
    run_seconds = (run_end - run_start) / sampling_rate_hz

    finite = np.isfinite(run_value)
    real = finite & (run_seconds <= run_value * gap_factor)
    if real.sum() < min_beats:
        return np.full(1, np.nan)

    beats = []
    for value, is_real in zip(run_value, real, strict=True):
        if is_real:
            beats.append(value)
        elif not beats or np.isfinite(beats[-1]):
            beats.append(np.nan)
    return np.asarray(beats, dtype=float)
