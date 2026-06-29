#!/usr/bin/env python3
"""
MPCA analysis pipeline for bi-analyte single-molecule SERS data.

Version: v5 with requested x-axis range cropping, fixed-length resampling,
optional exclusion of labels such as None from the PCA fitting stage,
selectable smoothing before normalization (Savitzky-Golay or moving average),
and soft min-max normalization.

This script loads classified Raman/SERS spectra from a folder tree, performs
Modified PCA-like two-component unmixing, and exports publication-quality figures
and CSV tables for single-molecule sensitivity evidence.

Expected data layout example:
  data/r6g-4-MPY-mix(1089)/
    (R6G-13)(4-MPY-11)/
      231-R6G/*.csv or merged.csv
      9-4-MPY/*.csv or merged.csv
      1-mix/*.csv or merged.csv
      848-none/*.csv or merged.csv

Single-spectrum CSV files: first column = Raman shift / wavelength grid,
second column = intensity.
Merged CSV files: first column = grid, following columns = spectra.

Author: ChatGPT
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    sparse = None
    spsolve = None
    savgol_filter = None


LABEL_ORDER = ["R6G", "4-MPY", "Mix", "None"]
LABEL_COLORS = {
    "R6G": "#d62728",
    "4-MPY": "#2ca02c",
    "Mix": "#1f77b4",
    "None": "#222222",
}
LABEL_MARKERS = {
    "R6G": "o",
    "4-MPY": "o",
    "Mix": "o",
    "None": "o",
}
DEFAULT_R6G_PEAKS = [613, 775, 1187, 1309, 1360, 1506, 1569, 1648]
DEFAULT_MPY_PEAKS = [1008, 1098, 1207, 1575, 1600]


@dataclass
class SpectrumRecord:
    x: np.ndarray
    y: np.ndarray
    label: str
    group: str
    terminal_dir: str
    source_file: str
    source_column: str


@dataclass
class LoadedDataset:
    x: np.ndarray
    spectra: np.ndarray  # shape: T x N
    metadata: pd.DataFrame
    preprocessing_info: Dict[str, object] = field(default_factory=dict)


@dataclass
class MPCAResult:
    mean_spectrum: np.ndarray
    components: np.ndarray        # 2 x N
    raw_scores: np.ndarray        # T x 2, PC scores before analyte-axis transform
    scores: np.ndarray            # T x 2, alpha/beta coefficients
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    basis_vectors_raw_score_space: np.ndarray  # columns = pure R6G and pure 4-MPY directions
    probability: np.ndarray       # T, NaN for invalid/None if computed with mask later
    pca_training_mask: np.ndarray  # T, True for spectra used to fit PCA eigenvectors/mean
    pca_excluded_labels: List[str]


def infer_label(name: str) -> Optional[str]:
    """Infer class label from a terminal directory name."""
    s = name.lower()
    s_clean = re.sub(r"[\s_]+", "-", s)
    if "none" in s_clean or re.search(r"(^|[-()])non($|[-()0-9])", s_clean):
        return "None"
    if "mix" in s_clean or "mixed" in s_clean:
        return "Mix"
    if "4-mpy" in s_clean or "4mpy" in s_clean or re.search(r"(^|[-()])mpy($|[-()0-9])", s_clean):
        return "4-MPY"
    if "r6g" in s_clean or "r-6g" in s_clean:
        return "R6G"
    return None


def sanitize_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:150] if len(text) > 150 else text


def parse_float_list(text: Optional[str]) -> List[float]:
    if not text:
        return []
    parts = re.split(r"[,;\s]+", text.strip())
    return [float(p) for p in parts if p]


def parse_label_list(text: Optional[str]) -> List[str]:
    """Parse comma/semicolon/space-separated labels and normalize common aliases."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw_parts = [str(v) for v in text]
    else:
        raw_parts = re.split(r"[,;\s]+", str(text).strip())
    aliases = {
        "r6g": "R6G",
        "r-6g": "R6G",
        "4-mpy": "4-MPY",
        "4mpy": "4-MPY",
        "mpy": "4-MPY",
        "mix": "Mix",
        "mixed": "Mix",
        "none": "None",
        "non": "None",
        "null": "None",
    }
    labels: List[str] = []
    for part in raw_parts:
        key = str(part).strip()
        if not key:
            continue
        norm = aliases.get(key.lower(), key)
        if norm not in LABEL_ORDER:
            raise argparse.ArgumentTypeError(
                f"Unknown label {key!r}; allowed labels are: {', '.join(LABEL_ORDER)}"
            )
        if norm not in labels:
            labels.append(norm)
    return labels


def setup_matplotlib(font_size: float = 8.0) -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.linewidth": 0.8,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_csv_numeric(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Read CSV-like file with or without header; return numeric array and header row if present."""
    try:
        raw = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    except Exception as exc:
        raise ValueError(f"Could not read CSV {path}: {exc}") from exc

    if raw.empty:
        raise ValueError(f"Empty CSV file: {path}")

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    first_row = numeric.iloc[0]
    header_labels: Optional[List[str]] = None

    # Header row if the first row is mostly non-numeric or the first cell is not numeric.
    first_numeric_fraction = float(first_row.notna().mean())
    if pd.isna(first_row.iloc[0]) or first_numeric_fraction < 0.75:
        header_labels = [str(v) for v in raw.iloc[0].tolist()]
        numeric = numeric.iloc[1:].reset_index(drop=True)

    numeric = numeric.dropna(axis=1, how="all")
    # Drop rows with missing x and rows that contain no intensity values.
    if numeric.shape[1] < 2:
        raise ValueError(f"CSV file has fewer than two numeric columns: {path}")
    numeric = numeric[numeric.iloc[:, 0].notna()]
    numeric = numeric.dropna(axis=0, how="all")

    arr = numeric.to_numpy(dtype=float)
    return arr, header_labels


def clean_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 3:
        raise ValueError("Spectrum has fewer than 3 valid points")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Average duplicated x values.
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        y_sum = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_sum, inverse, y)
        np.add.at(counts, inverse, 1)
        y = y_sum / np.maximum(counts, 1)
        x = unique_x
    return x, y


def read_spectra_from_dir(data_dir: Path, label: str, group: str) -> List[SpectrumRecord]:
    files = sorted([p for p in data_dir.glob("*.csv") if not p.name.startswith(".")])
    if not files:
        return []

    merged_candidates = [p for p in files if p.name.lower() == "merged.csv"]
    records: List[SpectrumRecord] = []

    if merged_candidates:
        merged = merged_candidates[0]
        arr, header = read_csv_numeric(merged)
        x = arr[:, 0]
        for j in range(1, arr.shape[1]):
            y = arr[:, j]
            try:
                x_clean, y_clean = clean_xy(x, y)
            except ValueError:
                continue
            if header and j < len(header):
                col_name = header[j]
            else:
                col_name = f"col_{j}"
            records.append(SpectrumRecord(
                x=x_clean,
                y=y_clean,
                label=label,
                group=group,
                terminal_dir=str(data_dir),
                source_file=str(merged),
                source_column=col_name,
            ))
        return records

    for f in files:
        arr, header = read_csv_numeric(f)
        if arr.shape[1] < 2:
            continue
        # For ordinary files, use first two numeric columns.
        x, y = arr[:, 0], arr[:, 1]
        try:
            x_clean, y_clean = clean_xy(x, y)
        except ValueError:
            continue
        records.append(SpectrumRecord(
            x=x_clean,
            y=y_clean,
            label=label,
            group=group,
            terminal_dir=str(data_dir),
            source_file=str(f),
            source_column=header[1] if header and len(header) > 1 else "intensity",
        ))
    return records


def discover_spectra(root: Path) -> List[SpectrumRecord]:
    root = root.resolve()
    records: List[SpectrumRecord] = []
    for dirpath, _, filenames in os_walk_sorted(root):
        csv_files = [f for f in filenames if f.lower().endswith(".csv")]
        if not csv_files:
            continue
        label = infer_label(dirpath.name)
        if label is None:
            continue
        group = dirpath.parent.name if dirpath.parent != root.parent else root.name
        records.extend(read_spectra_from_dir(dirpath, label, group))
    return records


def os_walk_sorted(root: Path):
    """Like os.walk but sorted and Path-friendly."""
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield Path(dirpath), dirnames, filenames


def apply_x_range(records: List[SpectrumRecord], x_min: Optional[float], x_max: Optional[float]) -> List[SpectrumRecord]:
    if x_min is None and x_max is None:
        return records
    out: List[SpectrumRecord] = []
    for rec in records:
        mask = np.ones_like(rec.x, dtype=bool)
        if x_min is not None:
            mask &= rec.x >= x_min
        if x_max is not None:
            mask &= rec.x <= x_max
        if int(mask.sum()) >= 3:
            out.append(SpectrumRecord(
                x=rec.x[mask], y=rec.y[mask], label=rec.label, group=rec.group,
                terminal_dir=rec.terminal_dir, source_file=rec.source_file, source_column=rec.source_column,
            ))
    return out


def make_common_grid(
    records: List[SpectrumRecord],
    grid_mode: str = "intersection",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    n_points: Optional[int] = None,
    strict_range: bool = True,
) -> np.ndarray:
    """Construct the common x-axis grid used for all spectra.

    v2 behavior:
      * If n_points is provided, the script builds an evenly spaced fixed-length
        grid with np.linspace(lower, upper, n_points). This is the recommended
        mode when different CSV files have slightly different sampling.
      * If x_min/x_max are provided, lower/upper are exactly those requested
        values when they are covered by all spectra. This avoids silent
        extrapolation at the edges.
      * Without n_points, the original v1 behavior is preserved: use either the
        first spectrum grid or the intersection of all spectra after range
        filtering.
    """
    if not records:
        raise ValueError("No spectra records available")

    global_common_min = max(float(np.min(r.x)) for r in records)
    global_common_max = min(float(np.max(r.x)) for r in records)
    if not global_common_min < global_common_max:
        raise ValueError("Spectra do not share an overlapping x-axis range")

    lower = float(x_min) if x_min is not None else global_common_min
    upper = float(x_max) if x_max is not None else global_common_max
    if not lower < upper:
        raise ValueError(f"Invalid x-axis range: lower={lower}, upper={upper}")

    # Do not allow np.interp to silently extrapolate unless the user explicitly
    # disables strict range checking. A tiny tolerance accounts for floating
    # point roundoff in CSV parsing.
    tol = 1e-9 * max(abs(global_common_min), abs(global_common_max), abs(lower), abs(upper), 1.0)
    if strict_range and (lower < global_common_min - tol or upper > global_common_max + tol):
        raise ValueError(
            "Requested x-axis range is not covered by every spectrum. "
            f"Requested [{lower:g}, {upper:g}], common coverage "
            f"[{global_common_min:g}, {global_common_max:g}]. "
            "Use a narrower range or pass --allow-edge-extrapolation."
        )
    if not strict_range:
        # Still keep a valid monotonic range; values outside individual spectra
        # will be filled by np.interp edge values. This option is not recommended
        # for formal analysis.
        pass

    if n_points is not None:
        if int(n_points) < 3:
            raise ValueError("--n-points must be >= 3")
        return np.linspace(lower, upper, int(n_points), dtype=float)

    ref_x = records[0].x.copy()
    if grid_mode == "first":
        grid = ref_x[(ref_x >= lower) & (ref_x <= upper)]
        if grid.size < 3:
            raise ValueError("First-spectrum x-axis grid has fewer than 3 points inside the requested range")
        return grid

    # Original intersection behavior, but constrained by optional x_min/x_max.
    common_min = max(global_common_min, lower)
    common_max = min(global_common_max, upper)
    if not common_min < common_max:
        raise ValueError("Spectra do not share an overlapping x-axis range after filtering")
    grid = ref_x[(ref_x >= common_min) & (ref_x <= common_max)]
    if grid.size < 3:
        # Fallback: use shortest grid inside the common range.
        candidate = min(records, key=lambda r: r.x.size).x
        grid = candidate[(candidate >= common_min) & (candidate <= common_max)]
    if grid.size < 3:
        raise ValueError("Common x-axis grid has fewer than 3 points")
    return grid


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    if sparse is None or spsolve is None:
        raise RuntimeError("scipy is required for ALS baseline correction")
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 5:
        return np.zeros_like(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    w = np.ones(n)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return np.asarray(z, dtype=float)


def canonical_normalize_mode(normalize: str) -> str:
    mode = str(normalize or "none").strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "max": "max",
        "vector": "vector",
        "area": "area",
        "snv": "snv",
        "minmax": "minmax",
        "min_max": "minmax",
        "softminmax": "soft_minmax",
        "soft_minmax": "soft_minmax",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown normalization mode: {normalize}")
    return aliases[mode]


def canonical_smoothing_method(method: str) -> str:
    """Normalize smoothing method aliases."""
    mode = str(method or "sg").strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "disable": "none",
        "disabled": "none",
        "sg": "sg",
        "s_g": "sg",
        "savgol": "sg",
        "savitzky_golay": "sg",
        "savitzkygolay": "sg",
        "moving_average": "moving_average",
        "movingavg": "moving_average",
        "moving_mean": "moving_average",
        "mean": "moving_average",
        "ma": "moving_average",
        "boxcar": "moving_average",
        "rolling_mean": "moving_average",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown smoothing method: {method}")
    return aliases[mode]


def apply_smoothing_if_requested(
    y: np.ndarray,
    smooth_method: str = "sg",
    smooth_window: int = 0,
    smooth_poly: int = 3,
) -> np.ndarray:
    """Apply one smoothing method with the same parameters to every spectrum.

    Smoothing is intentionally performed before normalization. A window value of
    0 disables smoothing regardless of the selected method.
    """
    y = np.asarray(y, dtype=float).copy()
    method = canonical_smoothing_method(smooth_method)
    window = int(smooth_window or 0)
    if method == "none" or window <= 1:
        return y

    if method == "sg":
        if window < 3:
            return y
        if savgol_filter is None:
            raise RuntimeError("scipy is required for Savitzky-Golay smoothing")
        if window % 2 == 0:
            window += 1
        if window >= y.size:
            # Savitzky-Golay requires window_length <= signal length and odd.
            window = y.size if y.size % 2 == 1 else y.size - 1
        if window >= 3:
            poly = min(int(smooth_poly), int(window) - 1)
            y = savgol_filter(y, window_length=int(window), polyorder=int(poly))
        return y

    if method == "moving_average":
        # Use a centered boxcar average and edge-value padding so the output
        # length is identical to the input length. Odd windows give symmetric
        # smoothing; even windows are rounded up to the next odd value.
        if window < 2:
            return y
        if window % 2 == 0:
            window += 1
        window = min(window, y.size)
        if window <= 1:
            return y
        kernel = np.ones(int(window), dtype=float) / float(window)
        pad_left = int(window) // 2
        pad_right = int(window) - 1 - pad_left
        y_pad = np.pad(y, (pad_left, pad_right), mode="edge")
        return np.convolve(y_pad, kernel, mode="valid")

    raise ValueError(f"Unknown smoothing method: {smooth_method}")


def preprocess_before_normalization(
    y: np.ndarray,
    baseline: str = "none",
    baseline_lam: float = 1e5,
    baseline_p: float = 0.01,
    smooth_method: str = "sg",
    smooth_window: int = 0,
    smooth_poly: int = 3,
) -> np.ndarray:
    """Apply optional baseline correction and optional smoothing before normalization.

    v5 deliberately separates this step from normalization so that normalization
    constants, including the soft min-max constant derived from None spectra, are
    computed from spectra that have already received the same baseline/smoothing
    treatment but have not yet been normalized.
    """
    y = np.asarray(y, dtype=float).copy()

    if baseline == "als":
        y = y - baseline_als(y, lam=baseline_lam, p=baseline_p)
    elif baseline == "poly1":
        x = np.arange(y.size, dtype=float)
        coeff = np.polyfit(x, y, deg=1)
        y = y - np.polyval(coeff, x)
    elif baseline == "poly2":
        x = np.arange(y.size, dtype=float)
        coeff = np.polyfit(x, y, deg=2)
        y = y - np.polyval(coeff, x)
    elif baseline != "none":
        raise ValueError(f"Unknown baseline method: {baseline}")

    y = apply_smoothing_if_requested(y, smooth_method=smooth_method, smooth_window=smooth_window, smooth_poly=smooth_poly)
    return y


def normalize_spectrum(
    y: np.ndarray,
    normalize: str = "none",
    soft_minmax_constant: float = 0.0,
) -> np.ndarray:
    """Normalize one preprocessed spectrum.

    soft_minmax uses (max - min + c) in the denominator, where c is normally
    multiplier * std(None spectra) and is shared by all spectra in the analysis.
    """
    y = np.asarray(y, dtype=float).copy()
    mode = canonical_normalize_mode(normalize)

    if mode == "max":
        denom = np.nanmax(np.abs(y))
        if denom > 0:
            y = y / denom
    elif mode == "minmax":
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        denom = ymax - ymin
        if np.isfinite(denom) and denom > 0:
            y = (y - ymin) / denom
        else:
            y = np.zeros_like(y)
    elif mode == "soft_minmax":
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        c = float(soft_minmax_constant)
        if not np.isfinite(c) or c < 0:
            raise ValueError(f"soft_minmax_constant must be finite and non-negative, got {soft_minmax_constant}")
        denom = (ymax - ymin) + c
        if np.isfinite(denom) and denom > 0:
            y = (y - ymin) / denom
        else:
            y = np.zeros_like(y)
    elif mode == "vector":
        denom = np.linalg.norm(y)
        if denom > 0:
            y = y / denom
    elif mode == "area":
        denom = np.trapz(np.abs(y))
        if denom > 0:
            y = y / denom
    elif mode == "snv":
        std = np.nanstd(y)
        if std > 0:
            y = (y - np.nanmean(y)) / std
    elif mode != "none":
        raise ValueError(f"Unknown normalization mode: {normalize}")
    return y


def preprocess_spectrum(
    y: np.ndarray,
    baseline: str = "none",
    baseline_lam: float = 1e5,
    baseline_p: float = 0.01,
    smooth_method: str = "sg",
    smooth_window: int = 0,
    smooth_poly: int = 3,
    normalize: str = "none",
    soft_minmax_constant: float = 0.0,
) -> np.ndarray:
    y = preprocess_before_normalization(
        y,
        baseline=baseline,
        baseline_lam=baseline_lam,
        baseline_p=baseline_p,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
    )
    y = normalize_spectrum(y, normalize=normalize, soft_minmax_constant=soft_minmax_constant)
    return y

def load_dataset(
    root: Path,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    grid_mode: str = "intersection",
    n_points: Optional[int] = None,
    allow_edge_extrapolation: bool = False,
    baseline: str = "none",
    baseline_lam: float = 1e5,
    baseline_p: float = 0.01,
    smooth_method: str = "sg",
    smooth_window: int = 0,
    smooth_poly: int = 3,
    normalize: str = "none",
    soft_minmax_none_std_multiplier: float = 10.0,
) -> LoadedDataset:
    records = discover_spectra(root)
    if not records:
        raise ValueError(f"No labelled spectra were found under: {root}")

    # v2/v3/v4: if --n-points is given, construct the requested fixed-length grid
    # directly from the original spectra and interpolate all spectra onto it.
    # We do not crop first, because cropping before interpolation can remove
    # points just outside the requested range and create artificial edge values.
    if n_points is not None:
        grid = make_common_grid(
            records,
            grid_mode=grid_mode,
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
            strict_range=not allow_edge_extrapolation,
        )
    else:
        records = apply_x_range(records, x_min, x_max)
        if not records:
            raise ValueError("No spectra remain after x-axis range filtering")
        grid = make_common_grid(
            records,
            grid_mode=grid_mode,
            x_min=x_min,
            x_max=x_max,
            n_points=None,
            strict_range=not allow_edge_extrapolation,
        )

    # First pass: interpolation + optional baseline correction + optional smoothing.
    # Normalization is intentionally delayed until all spectra are available, because
    # v5 soft min-max normalization needs one shared constant derived from None spectra.
    pre_norm_list: List[np.ndarray] = []
    meta_rows = []
    for i, rec in enumerate(records):
        y_interp = np.interp(grid, rec.x, rec.y)
        y_pre = preprocess_before_normalization(
            y_interp,
            baseline=baseline,
            baseline_lam=baseline_lam,
            baseline_p=baseline_p,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
        )
        pre_norm_list.append(y_pre)
        meta_rows.append({
            "event_id": i,
            "label": rec.label,
            "group": rec.group,
            "terminal_dir": rec.terminal_dir,
            "source_file": rec.source_file,
            "source_column": rec.source_column,
        })

    pre_norm_spectra = np.vstack(pre_norm_list)
    metadata = pd.DataFrame(meta_rows)
    normalize_mode = canonical_normalize_mode(normalize)

    soft_minmax_none_std = None
    soft_minmax_constant = 0.0
    if normalize_mode == "soft_minmax":
        mult = float(soft_minmax_none_std_multiplier)
        if not np.isfinite(mult) or mult < 0:
            raise ValueError("--soft-minmax-none-std-multiplier must be finite and non-negative")
        none_mask = metadata["label"].to_numpy() == "None"
        if not np.any(none_mask):
            raise ValueError(
                "soft_minmax normalization requires at least one spectrum labelled None "
                "to estimate the shared noise standard deviation."
            )
        soft_minmax_none_std = float(np.nanstd(pre_norm_spectra[none_mask]))
        if not np.isfinite(soft_minmax_none_std):
            raise ValueError("Could not compute a finite None-spectrum standard deviation for soft_minmax normalization")
        soft_minmax_constant = mult * soft_minmax_none_std

    spectra_list = [
        normalize_spectrum(y, normalize=normalize_mode, soft_minmax_constant=soft_minmax_constant)
        for y in pre_norm_spectra
    ]
    spectra = np.vstack(spectra_list)

    preprocessing_info = {
        "baseline": baseline,
        "baseline_lam": float(baseline_lam),
        "baseline_p": float(baseline_p),
        "smoothing_method": canonical_smoothing_method(smooth_method),
        "smoothing_enabled": bool(canonical_smoothing_method(smooth_method) != "none" and smooth_window and smooth_window > 1),
        "smoothing_window_requested": int(smooth_window),
        "savgol_polyorder": int(smooth_poly),
        "savgol_enabled": bool(canonical_smoothing_method(smooth_method) == "sg" and smooth_window and smooth_window >= 3),
        "moving_average_enabled": bool(canonical_smoothing_method(smooth_method) == "moving_average" and smooth_window and smooth_window > 1),
        "normalize": normalize_mode,
        "soft_minmax_none_std_multiplier": float(soft_minmax_none_std_multiplier),
        "soft_minmax_none_std": soft_minmax_none_std,
        "soft_minmax_constant": float(soft_minmax_constant),
    }
    return LoadedDataset(x=grid, spectra=spectra, metadata=metadata, preprocessing_info=preprocessing_info)

def choose_component_count(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return components, raw scores, eigenvalues from centered matrix via SVD."""
    # Xc = U S Vt. Eigenvalues of covariance = S^2 / (T-1).
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    denom = max(Xc.shape[0] - 1, 1)
    eigenvalues = (S ** 2) / denom
    components = Vt[:2, :]
    raw_scores = Xc @ components.T
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return components, raw_scores, eigenvalues, explained


def robust_centroid(scores: np.ndarray, labels: np.ndarray, label: str) -> np.ndarray:
    mask = labels == label
    if not np.any(mask):
        raise ValueError(f"Cannot find spectra labelled {label!r}; required for analyte-axis calibration")
    return np.median(scores[mask], axis=0)


def orient_and_transform_scores(raw_scores: np.ndarray, labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Transform PCA scores into analyte coefficients alpha/beta using pure-label axes.

    The first two PCA scores are a two-dimensional plane. We estimate the R6G and
    4-MPY rays from robust centroids of the pure classes, optionally subtracting the
    null-event centroid so that null events cluster around the origin. Coefficients are
    then coordinates in the basis [R6G_axis, MPY_axis].
    """
    labels_arr = np.asarray(labels)
    score0 = raw_scores.copy()

    if np.any(labels_arr == "None"):
        origin = robust_centroid(score0, labels_arr, "None")
        score0 = score0 - origin

    u_r6g = robust_centroid(score0, labels_arr, "R6G")
    u_mpy = robust_centroid(score0, labels_arr, "4-MPY")

    # If a centroid is very close to zero, use the mean as fallback.
    for lab, vec in [("R6G", u_r6g), ("4-MPY", u_mpy)]:
        if np.linalg.norm(vec) < 1e-12:
            mask = labels_arr == lab
            replacement = np.mean(score0[mask], axis=0)
            if lab == "R6G":
                u_r6g = replacement
            else:
                u_mpy = replacement

    B = np.column_stack([u_r6g, u_mpy])  # 2 x 2
    det = float(np.linalg.det(B))
    if abs(det) < 1e-12:
        # Fallback to raw PC axes if analyte centroids are almost collinear.
        B = np.eye(2)
    coeff = score0 @ np.linalg.inv(B).T

    # Normalize scale so pure-class median alpha/beta are near the same order.
    # This makes p = alpha/(alpha+beta) more stable when one dye has much larger score magnitude.
    labels_arr = np.asarray(labels)
    a_med = np.nanmedian(coeff[labels_arr == "R6G", 0]) if np.any(labels_arr == "R6G") else 1.0
    b_med = np.nanmedian(coeff[labels_arr == "4-MPY", 1]) if np.any(labels_arr == "4-MPY") else 1.0
    scale = np.array([a_med if abs(a_med) > 1e-12 else 1.0,
                      b_med if abs(b_med) > 1e-12 else 1.0], dtype=float)
    coeff = coeff / scale

    # Flip if necessary to keep pure signals positive.
    if np.nanmedian(coeff[labels_arr == "R6G", 0]) < 0:
        coeff[:, 0] *= -1
        B[:, 0] *= -1
    if np.nanmedian(coeff[labels_arr == "4-MPY", 1]) < 0:
        coeff[:, 1] *= -1
        B[:, 1] *= -1

    return coeff, B


def run_mpca(
    dataset: LoadedDataset,
    clip_probability_coefficients: bool = True,
    pca_exclude_labels: Optional[Sequence[str]] = None,
) -> MPCAResult:
    """Fit the two MPCA/PCA directions, then project every spectrum.

    By default, all labels are used to fit the PCA mean spectrum and eigenvectors,
    matching v1/v2 behavior. If pca_exclude_labels contains labels such as "None",
    those spectra are excluded only from PCA fitting. They are still projected onto
    the fitted components and remain visible in scatter/QC outputs.
    """
    X = np.asarray(dataset.spectra, dtype=float)
    labels = dataset.metadata["label"].to_numpy()
    excluded = list(pca_exclude_labels or [])

    train_mask = np.ones(labels.shape[0], dtype=bool)
    for lab in excluded:
        train_mask &= labels != lab

    n_train = int(np.sum(train_mask))
    if n_train < 3:
        raise ValueError(
            f"Only {n_train} spectra remain for PCA fitting after excluding labels {excluded}. "
            "At least 3 spectra are recommended/required for stable two-component MPCA."
        )
    if X.shape[1] < 2:
        raise ValueError("At least 2 spectral points are required for two-component MPCA.")

    # PCA is fitted only on the selected training spectra. All spectra, including
    # excluded labels, are then centered by the training mean and projected into
    # the same two-component space.
    X_train = X[train_mask]
    mean_spectrum = np.mean(X_train, axis=0)
    Xc_train = X_train - mean_spectrum
    components, _train_raw_scores, eigenvalues, explained = choose_component_count(Xc_train)
    if components.shape[0] < 2:
        raise ValueError(
            "PCA fitting produced fewer than two components. Check that enough non-excluded "
            "spectra and spectral points are available."
        )
    components = components[:2, :]
    raw_scores = (X - mean_spectrum) @ components.T
    scores, basis = orient_and_transform_scores(raw_scores, labels)

    alpha = scores[:, 0].copy()
    beta = scores[:, 1].copy()
    if clip_probability_coefficients:
        alpha_eff = np.clip(alpha, 0, None)
        beta_eff = np.clip(beta, 0, None)
    else:
        alpha_eff = alpha
        beta_eff = beta
    denom = alpha_eff + beta_eff
    probability = np.full(alpha.shape, np.nan, dtype=float)
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    probability[valid] = alpha_eff[valid] / denom[valid]
    return MPCAResult(
        mean_spectrum=mean_spectrum,
        components=components,
        raw_scores=raw_scores,
        scores=scores,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        basis_vectors_raw_score_space=basis,
        probability=probability,
        pca_training_mask=train_mask,
        pca_excluded_labels=excluded,
    )


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig: plt.Figure, outbase: Path, dpi: int = 300) -> None:
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def nearest_peak_indices(x: np.ndarray, peaks: Sequence[float], tolerance: Optional[float] = None) -> List[Tuple[float, int]]:
    if x.size < 2:
        return []
    dx = float(np.nanmedian(np.diff(np.sort(x))))
    tol = tolerance if tolerance is not None else max(5.0, 3.0 * abs(dx))
    out = []
    for p in peaks:
        idx = int(np.argmin(np.abs(x - p)))
        if abs(float(x[idx]) - p) <= tol:
            out.append((float(p), idx))
    return out


def typical_spectrum_indices(dataset: LoadedDataset) -> Dict[str, int]:
    labels = dataset.metadata["label"].to_numpy()
    X = dataset.spectra
    out: Dict[str, int] = {}
    for lab in LABEL_ORDER:
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        class_X = X[idx]
        med = np.median(class_X, axis=0)
        # Scale-insensitive distance to class median.
        denom = np.linalg.norm(class_X, axis=1) * max(np.linalg.norm(med), 1e-12)
        corr = (class_X @ med) / np.maximum(denom, 1e-12)
        chosen = idx[int(np.nanargmax(corr))]
        out[lab] = int(chosen)
    return out


def scaled_for_plot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y0 = y - np.nanpercentile(y, 5)
    scale = np.nanpercentile(y0, 99) - np.nanpercentile(y0, 1)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(y0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return y0 / scale


def plot_representative_spectra(
    dataset: LoadedDataset,
    outdir: Path,
    x_label: str,
    r6g_peaks: Sequence[float],
    mpy_peaks: Sequence[float],
    dpi: int,
) -> None:
    idx_map = typical_spectrum_indices(dataset)
    rows = []
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    offset_step = 1.15
    plot_order = [lab for lab in ["None", "R6G", "4-MPY", "Mix"] if lab in idx_map]

    for k, lab in enumerate(plot_order):
        idx = idx_map[lab]
        y_plot = scaled_for_plot(dataset.spectra[idx]) + k * offset_step
        ax.plot(dataset.x, y_plot, lw=1.2, color=LABEL_COLORS[lab], label=lab)
        ax.text(dataset.x[-1], k * offset_step + 0.15, lab, color=LABEL_COLORS[lab],
                ha="right", va="bottom", fontsize=7)
        row = pd.DataFrame({
            "x": dataset.x,
            f"{lab}_event_{idx}": dataset.spectra[idx],
            f"{lab}_event_{idx}_scaled_offset": y_plot,
        })
        rows.append(row)

    # Mark characteristic peaks using small vertical ticks and labels.
    y_top = max((len(plot_order) - 1) * offset_step + 1.0, 1.0)
    for peak, idx in nearest_peak_indices(dataset.x, r6g_peaks):
        ax.axvline(dataset.x[idx], ymin=0.02, ymax=0.10, color=LABEL_COLORS["R6G"], lw=0.7, alpha=0.8)
        ax.text(dataset.x[idx], y_top, f"{peak:g}", rotation=90, ha="center", va="top",
                fontsize=6, color=LABEL_COLORS["R6G"])
    for peak, idx in nearest_peak_indices(dataset.x, mpy_peaks):
        ax.axvline(dataset.x[idx], ymin=0.12, ymax=0.20, color=LABEL_COLORS["4-MPY"], lw=0.7, alpha=0.8)
        ax.text(dataset.x[idx], y_top - 0.25, f"{peak:g}", rotation=90, ha="center", va="top",
                fontsize=6, color=LABEL_COLORS["4-MPY"])

    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized intensity + offset")
    ax.set_title("Representative SERS spectra")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    savefig(fig, outdir / "figure_1_representative_stacked_spectra", dpi=dpi)

    if rows:
        merged = rows[0][["x"]].copy()
        for row in rows:
            for col in row.columns:
                if col != "x":
                    merged[col] = row[col].values
        merged.to_csv(outdir / "figure_1_representative_stacked_spectra_data.csv", index=False)


def plot_event_counts(dataset: LoadedDataset, outdir: Path, dpi: int) -> None:
    counts = dataset.metadata["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    counts_df = counts.rename_axis("label").reset_index(name="count")
    counts_df.to_csv(outdir / "figure_2_event_counts_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    labels = counts.index.tolist()
    values = counts.values.astype(int)
    colors = [LABEL_COLORS[l] for l in labels]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ymax = max(values.max() * 1.18 if values.size else 1, 1)
    ax.set_ylim(0, ymax)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.015, str(int(val)),
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of events")
    ax.set_title("Event occurrences")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, outdir / "figure_2_event_counts", dpi=dpi)


def plot_mpca_scatter(dataset: LoadedDataset, result: MPCAResult, outdir: Path, dpi: int) -> None:
    labels = dataset.metadata["label"].to_numpy()
    scores = result.scores
    score_df = dataset.metadata.copy()
    score_df["alpha"] = scores[:, 0]
    score_df["beta"] = scores[:, 1]
    score_df["raw_pc1"] = result.raw_scores[:, 0]
    score_df["raw_pc2"] = result.raw_scores[:, 1]
    score_df["probability_r6g"] = result.probability
    score_df["used_for_pca_training"] = result.pca_training_mask
    score_df.to_csv(outdir / "figure_3_mpca_scores_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.1, 2.7))
    for lab in LABEL_ORDER:
        mask = labels == lab
        if not np.any(mask):
            continue
        alpha = 0.75 if lab != "None" else 0.35
        size = 8 if lab != "None" else 5
        ax.scatter(scores[mask, 0], scores[mask, 1], s=size, c=LABEL_COLORS[lab],
                   label=lab, marker=LABEL_MARKERS[lab], edgecolors="black" if lab != "None" else "none",
                   linewidths=0.25, alpha=alpha)
    ax.axhline(0, color="0.75", lw=0.7, zorder=0)
    ax.axvline(0, color="0.75", lw=0.7, zorder=0)
    ax.set_xlabel(r"$\alpha$ (R6G coefficient)")
    ax.set_ylabel(r"$\beta$ (4-MPY coefficient)")
    ax.set_title("MPCA coefficient plot")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, outdir / "figure_3_mpca_coefficient_scatter", dpi=dpi)


def plot_probability_histogram(dataset: LoadedDataset, result: MPCAResult, outdir: Path, bins: int, dpi: int) -> None:
    labels = dataset.metadata["label"].to_numpy()
    valid_event = (labels != "None") & np.isfinite(result.probability)
    p = result.probability[valid_event]
    p = p[(p >= -1e-9) & (p <= 1 + 1e-9)]
    p = np.clip(p, 0, 1)
    pd.DataFrame({"probability_r6g": p}).to_csv(outdir / "figure_4_probability_histogram_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    if p.size > 0:
        ax.hist(p, bins=np.linspace(0, 1, bins + 1), color="#ff7f0e", edgecolor="black", linewidth=0.7, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of R6G event")
    ax.set_ylabel("Number of events")
    ax.set_title("Single-molecule probability")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, outdir / "figure_4_probability_histogram", dpi=dpi)


def plot_qc_outputs(dataset: LoadedDataset, result: MPCAResult, outdir: Path, x_label: str, dpi: int) -> None:
    qcdir = ensure_outdir(outdir / "qc")

    eig_df = pd.DataFrame({
        "component": np.arange(1, result.eigenvalues.size + 1),
        "eigenvalue": result.eigenvalues,
        "explained_variance_ratio": result.explained_variance_ratio,
    })
    eig_df.to_csv(qcdir / "mpca_eigenvalues.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    n_show = min(50, result.eigenvalues.size)
    ax.plot(np.arange(1, n_show + 1), result.eigenvalues[:n_show], marker="o", ms=2, lw=1)
    ax.set_xlabel("Eigenvector #")
    ax.set_ylabel("Eigenvalue (a.u.)")
    ax.set_title("MPCA eigenvalues")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, qcdir / "qc_eigenvalues", dpi=dpi)

    labels = dataset.metadata["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(3.1, 2.7))
    for lab in LABEL_ORDER:
        mask = labels == lab
        if np.any(mask):
            ax.scatter(result.raw_scores[mask, 0], result.raw_scores[mask, 1], s=12,
                       c=LABEL_COLORS[lab], label=lab, alpha=0.7)
    ax.set_xlabel(r"raw $\alpha'$ / PC1 score")
    ax.set_ylabel(r"raw $\beta'$ / PC2 score")
    ax.set_title("Raw PCA score plot")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, qcdir / "qc_raw_pca_scores", dpi=dpi)

    # Supplementary-Fig-S16-like all spectra panels by class.
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.5), sharex=True)
    axes = axes.ravel()
    for ax, lab in zip(axes, LABEL_ORDER):
        mask = labels == lab
        idx = np.where(mask)[0]
        if idx.size == 0:
            ax.set_title(f"{lab}: 0 events")
            continue
        # Draw no more than 80 lines to keep vector files manageable; export all data separately.
        plot_idx = idx if idx.size <= 80 else idx[np.linspace(0, idx.size - 1, 80).astype(int)]
        for n, i in enumerate(plot_idx):
            y = scaled_for_plot(dataset.spectra[i]) + n * 0.03
            ax.plot(dataset.x, y, lw=0.35, color=LABEL_COLORS[lab], alpha=0.35)
        ax.set_title(f"{lab}: {idx.size} events")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[2:]:
        ax.set_xlabel(x_label)
    axes[0].set_ylabel("Intensity + offset")
    axes[2].set_ylabel("Intensity + offset")
    fig.tight_layout()
    savefig(fig, qcdir / "qc_all_spectra_by_label", dpi=dpi)

    # Eigenvectors / loadings.
    comp_df = pd.DataFrame({"x": dataset.x, "pc1": result.components[0], "pc2": result.components[1], "mean": result.mean_spectrum})
    comp_df.to_csv(qcdir / "mpca_components_and_mean.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(dataset.x, result.components[0], lw=1.0, label="PC1")
    ax.plot(dataset.x, result.components[1], lw=1.0, label="PC2")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loading")
    ax.set_title("First two MPCA loadings")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, qcdir / "qc_mpca_loadings", dpi=dpi)


def export_dataset_matrix(dataset: LoadedDataset, outdir: Path) -> None:
    matrix_df = pd.DataFrame(dataset.spectra.T, columns=[f"event_{i}" for i in range(dataset.spectra.shape[0])])
    matrix_df.insert(0, "x", dataset.x)
    matrix_df.to_csv(outdir / "cleaned_spectral_matrix_T_by_N_transposed.csv", index=False)
    dataset.metadata.to_csv(outdir / "cleaned_spectral_metadata.csv", index=False)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def summarize_analysis(dataset: LoadedDataset, result: MPCAResult, outdir: Path, args: argparse.Namespace) -> None:
    labels = dataset.metadata["label"].to_numpy()
    counts = dataset.metadata["label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    valid_p = result.probability[(labels != "None") & np.isfinite(result.probability)]
    summary = {
        "n_events": int(dataset.spectra.shape[0]),
        "n_points": int(dataset.spectra.shape[1]),
        "x_min": float(np.min(dataset.x)),
        "x_max": float(np.max(dataset.x)),
        "counts": {k: int(v) for k, v in counts.items()},
        "pca_excluded_labels": list(result.pca_excluded_labels),
        "pca_training_events": int(np.sum(result.pca_training_mask)),
        "pca_training_counts": {
            k: int(v) for k, v in dataset.metadata.loc[result.pca_training_mask, "label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict().items()
        },
        "explained_variance_ratio_pc1": float(result.explained_variance_ratio[0]) if result.explained_variance_ratio.size > 0 else None,
        "explained_variance_ratio_pc2": float(result.explained_variance_ratio[1]) if result.explained_variance_ratio.size > 1 else None,
        "probability_valid_events": int(valid_p.size),
        "probability_mean": float(np.nanmean(valid_p)) if valid_p.size else None,
        "probability_median": float(np.nanmedian(valid_p)) if valid_p.size else None,
        "preprocessing_info": _jsonable(dataset.preprocessing_info),
        "arguments": _jsonable(vars(args)),
    }
    with open(outdir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def analyze_one_root(root: Path, outdir: Path, args: argparse.Namespace) -> None:
    ensure_outdir(outdir)
    dataset = load_dataset(
        root,
        x_min=args.x_min,
        x_max=args.x_max,
        grid_mode=args.grid_mode,
        n_points=args.n_points,
        allow_edge_extrapolation=args.allow_edge_extrapolation,
        baseline=args.baseline,
        baseline_lam=args.baseline_lam,
        baseline_p=args.baseline_p,
        smooth_method=args.smooth_method,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        normalize=args.normalize,
        soft_minmax_none_std_multiplier=args.soft_minmax_none_std_multiplier,
    )
    result = run_mpca(
        dataset,
        clip_probability_coefficients=not args.no_clip_probability_coefficients,
        pca_exclude_labels=args.pca_exclude_labels,
    )

    export_dataset_matrix(dataset, outdir)
    plot_representative_spectra(dataset, outdir, args.x_label, args.r6g_peaks, args.mpy_peaks, args.dpi)
    plot_event_counts(dataset, outdir, args.dpi)
    plot_mpca_scatter(dataset, result, outdir, args.dpi)
    plot_probability_histogram(dataset, result, outdir, args.hist_bins, args.dpi)
    if args.qc:
        plot_qc_outputs(dataset, result, outdir, args.x_label, args.dpi)
    summarize_analysis(dataset, result, outdir, args)


def immediate_group_dirs(root: Path) -> List[Path]:
    groups = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MPCA pipeline for classified bi-analyte SERS spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Root folder containing classified spectra.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output folder for figures and CSV files.")
    parser.add_argument("--per-group", action="store_true", help="Analyze each immediate subfolder independently, plus a global analysis.")
    parser.add_argument("--global-only", action="store_true", help="Only analyze all spectra together, even when --per-group is set.")

    parser.add_argument("--x-min", type=float, default=None, help="Minimum x-axis value to keep.")
    parser.add_argument("--x-max", type=float, default=None, help="Maximum x-axis value to keep.")
    parser.add_argument("--x-range", nargs=2, type=float, metavar=("X_MIN", "X_MAX"), default=None, help="Convenience form for setting --x-min and --x-max, e.g. --x-range 500 1800.")
    parser.add_argument("--n-points", "--interp-points", "--resample-points", dest="n_points", type=int, default=None, help="If set, interpolate every spectrum onto a fixed-length linspace grid after range selection. Example: --x-range 500 1800 --n-points 1301.")
    parser.add_argument("--allow-edge-extrapolation", action="store_true", help="Allow edge extrapolation when the requested range is slightly outside some spectra. Not recommended for formal analysis.")
    parser.add_argument("--grid-mode", choices=["intersection", "first"], default="intersection", help="How to construct common x-axis grid when --n-points is not used.")
    parser.add_argument("--x-label", default="Raman shift (cm$^{-1}$)", help="x-axis label for plots.")

    parser.add_argument("--baseline", choices=["none", "als", "poly1", "poly2"], default="none", help="Baseline correction method.")
    parser.add_argument("--baseline-lam", type=float, default=1e5, help="ALS baseline smoothness parameter.")
    parser.add_argument("--baseline-p", type=float, default=0.01, help="ALS baseline asymmetry parameter.")
    parser.add_argument("--smooth-method", default="sg", help="Smoothing method before normalization: none, sg/Savitzky-Golay, or moving_average/ma/boxcar. A smoothing window of 0 disables smoothing.")
    parser.add_argument("--smooth-window", "--sg-window", "--savgol-window", dest="smooth_window", type=int, default=0, help="Smoothing window; 0 disables smoothing. For sg and moving_average, even windows are rounded up to the next odd value.")
    parser.add_argument("--smooth-poly", "--sg-poly", "--sg-polyorder", dest="smooth_poly", type=int, default=3, help="Savitzky-Golay polynomial order. Ignored when --smooth-method moving_average is used.")
    parser.add_argument("--normalize", choices=["none", "max", "minmax", "soft_minmax", "soft-minmax", "vector", "area", "snv"], default="none", help="Per-spectrum normalization before MPCA. soft_minmax uses (max-min+c) with c derived from None spectra.")
    parser.add_argument("--soft-minmax-none-std-multiplier", "--soft-minmax-k", dest="soft_minmax_none_std_multiplier", type=float, default=10.0, help="For --normalize soft_minmax, denominator constant c = this multiplier × std of all pre-normalized None spectra.")

    parser.add_argument("--hist-bins", type=int, default=20, help="Number of bins for probability histogram.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG figures.")
    parser.add_argument("--font-size", type=float, default=8.0, help="Base font size for figures.")
    parser.add_argument("--r6g-peaks", type=parse_float_list, default=DEFAULT_R6G_PEAKS, help="Comma-separated R6G peak positions to annotate.")
    parser.add_argument("--mpy-peaks", type=parse_float_list, default=DEFAULT_MPY_PEAKS, help="Comma-separated 4-MPY peak positions to annotate.")
    parser.add_argument("--qc", action="store_true", help="Export additional QC plots analogous to Supplementary Fig. S16.")
    parser.add_argument("--no-clip-probability-coefficients", action="store_true", help="Use raw alpha/beta values for p instead of clipping negatives to zero.")
    parser.add_argument(
        "--exclude-none-for-pca",
        action="store_true",
        help="Exclude spectra labelled None from the PCA fitting stage, then project them afterward for plotting/export."
    )
    parser.add_argument(
        "--pca-exclude-labels",
        type=parse_label_list,
        default=None,
        help="Comma/space-separated labels to exclude from PCA fitting only. Example: --pca-exclude-labels None or --pca-exclude-labels None,Mix."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.x_range is not None:
        args.x_min, args.x_max = float(args.x_range[0]), float(args.x_range[1])
    if args.n_points is not None and args.n_points < 3:
        parser.error("--n-points must be >= 3")
    try:
        args.smooth_method = canonical_smoothing_method(args.smooth_method)
    except ValueError as exc:
        parser.error(str(exc))
    if args.smooth_window is not None and args.smooth_window < 0:
        parser.error("--smooth-window must be >= 0")
    args.pca_exclude_labels = list(args.pca_exclude_labels or [])
    if args.exclude_none_for_pca and "None" not in args.pca_exclude_labels:
        args.pca_exclude_labels.append("None")
    setup_matplotlib(args.font_size)

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not data_dir.exists():
        parser.error(f"--data-dir does not exist: {data_dir}")
    ensure_outdir(out_dir)

    try:
        # Global analysis of all spectra.
        global_out = out_dir / "all_groups"
        analyze_one_root(data_dir, global_out, args)
        print(f"[OK] Global analysis saved to: {global_out}")

        if args.per_group and not args.global_only:
            for group_dir in immediate_group_dirs(data_dir):
                # Analyze only groups containing labelled data folders.
                try:
                    group_out = out_dir / sanitize_filename(group_dir.name)
                    analyze_one_root(group_dir, group_out, args)
                    print(f"[OK] Group analysis saved to: {group_out}")
                except Exception as exc:
                    print(f"[WARN] Skipped group {group_dir.name!r}: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
