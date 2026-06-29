#!/usr/bin/env python3
"""
Synthetic bi-analyte SERS dataset generator for the MPCA pipeline.

It creates a folder tree compatible with mpca_sers_analysis_v2/v3:

  synthetic_r6g_4mpy/
    synthetic_group_01/
      60-R6G/
        saved_col_000001.csv
        ...
        merged.csv
      60-4-MPY/
      20-Mix/
      400-None/

Single-spectrum CSV files contain two columns:
  Raman shift (cm-1), intensity

merged.csv contains:
  Raman shift (cm-1), saved_col_000001.csv, saved_col_000002.csv, ...

Synthetic assumptions:
  * R6G and 4-MPY each have fixed non-overlapping Lorentzian peaks.
  * Mix spectra are linear sums of the two templates.
  * None spectra contain only Gaussian noise plus a weak optional baseline.
  * Random amplitude variations are included so the test resembles real mapping data.

Author: ChatGPT
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_R6G_PEAKS = [612.0, 775.0, 1362.0, 1510.0, 1650.0]
DEFAULT_R6G_HEIGHTS = [1.00, 0.55, 0.85, 0.65, 0.50]
DEFAULT_R6G_WIDTHS = [5.0, 6.0, 7.0, 7.0, 8.0]

# Chosen deliberately away from the R6G synthetic peaks to make validation easy.
DEFAULT_MPY_PEAKS = [1008.0, 1110.0, 1245.0, 1585.0, 1720.0]
DEFAULT_MPY_HEIGHTS = [0.95, 0.70, 0.50, 0.85, 0.55]
DEFAULT_MPY_WIDTHS = [5.0, 5.5, 6.0, 7.0, 8.0]

LABEL_ORDER = ["R6G", "4-MPY", "Mix", "None"]


def parse_float_list(text: Optional[str], default: Sequence[float]) -> List[float]:
    if text is None or str(text).strip() == "":
        return list(default)
    parts = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(float(chunk))
    return parts


def lorentzian(x: np.ndarray, center: float, height: float, fwhm: float) -> np.ndarray:
    """Lorentzian peak with maximum equal to height and specified FWHM."""
    gamma = float(fwhm) / 2.0
    return height * (gamma * gamma) / ((x - center) ** 2 + gamma * gamma)


def make_template(
    x: np.ndarray,
    peaks: Sequence[float],
    heights: Sequence[float],
    widths: Sequence[float],
) -> np.ndarray:
    if not (len(peaks) == len(heights) == len(widths)):
        raise ValueError("peaks, heights, and widths must have the same length")
    y = np.zeros_like(x, dtype=float)
    for p, h, w in zip(peaks, heights, widths):
        y += lorentzian(x, float(p), float(h), float(w))
    maxv = float(np.max(y))
    if maxv > 0:
        y = y / maxv
    return y


def weak_baseline(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(x)
    xn = (x - np.mean(x)) / max(np.ptp(x), 1.0)
    offset = rng.normal(0.0, 0.2 * strength)
    slope = rng.normal(0.0, 0.5 * strength)
    curve = rng.normal(0.0, 0.3 * strength)
    return offset + slope * xn + curve * (xn ** 2)


def noisy_spectrum(
    x: np.ndarray,
    r6g_template: np.ndarray,
    mpy_template: np.ndarray,
    label: str,
    rng: np.random.Generator,
    signal_scale: float,
    noise_std: float,
    none_noise_std: float,
    baseline_strength: float,
    amp_cv: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Create one synthetic spectrum and return its ground-truth coefficients."""
    if label == "R6G":
        alpha = float(rng.lognormal(mean=0.0, sigma=amp_cv))
        beta = 0.0
    elif label == "4-MPY":
        alpha = 0.0
        beta = float(rng.lognormal(mean=0.0, sigma=amp_cv))
    elif label == "Mix":
        # Let mix events span different R6G/4-MPY ratios.
        alpha = float(rng.lognormal(mean=0.0, sigma=amp_cv))
        beta = float(rng.lognormal(mean=0.0, sigma=amp_cv))
    elif label == "None":
        alpha = 0.0
        beta = 0.0
    else:
        raise ValueError(f"Unknown label: {label}")

    y = signal_scale * (alpha * r6g_template + beta * mpy_template)
    sigma = none_noise_std if label == "None" else noise_std
    y = y + weak_baseline(x, rng, baseline_strength)
    y = y + rng.normal(0.0, sigma, size=x.shape)

    # A small positive detector offset keeps intensities realistic while PCA still
    # subtracts the mean spectrum later.
    detector_offset = max(0.0, 3.0 * sigma)
    y = y + detector_offset

    truth = {
        "true_r6g_alpha": alpha,
        "true_mpy_beta": beta,
        "true_probability_r6g": alpha / (alpha + beta) if (alpha + beta) > 0 else np.nan,
        "noise_std": sigma,
    }
    return y.astype(float), truth


def write_single_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    df = pd.DataFrame({"Raman shift (cm-1)": x, "intensity": y})
    df.to_csv(path, index=False)


def write_merged_csv(path: Path, x: np.ndarray, spectra: List[np.ndarray], names: List[str]) -> None:
    data = {"Raman shift (cm-1)": x}
    data.update({name: y for name, y in zip(names, spectra)})
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def generate_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_merged and args.no_single_files:
        raise ValueError("At least one output format is required; do not use both --no-merged and --no-single-files.")

    rng = np.random.default_rng(args.seed)
    x = np.linspace(args.x_start, args.x_end, args.n_points, dtype=float)

    r6g_peaks = parse_float_list(args.r6g_peaks, DEFAULT_R6G_PEAKS)
    r6g_heights = parse_float_list(args.r6g_heights, DEFAULT_R6G_HEIGHTS)
    r6g_widths = parse_float_list(args.r6g_widths, DEFAULT_R6G_WIDTHS)
    mpy_peaks = parse_float_list(args.mpy_peaks, DEFAULT_MPY_PEAKS)
    mpy_heights = parse_float_list(args.mpy_heights, DEFAULT_MPY_HEIGHTS)
    mpy_widths = parse_float_list(args.mpy_widths, DEFAULT_MPY_WIDTHS)

    r6g_template = make_template(x, r6g_peaks, r6g_heights, r6g_widths)
    mpy_template = make_template(x, mpy_peaks, mpy_heights, mpy_widths)

    counts = {
        "R6G": int(args.n_r6g),
        "4-MPY": int(args.n_mpy),
        "Mix": int(args.n_mix),
        "None": int(args.n_none),
    }
    folder_names = {
        "R6G": f"{counts['R6G']}-R6G",
        "4-MPY": f"{counts['4-MPY']}-4-MPY",
        "Mix": f"{counts['Mix']}-Mix",
        "None": f"{counts['None']}-None",
    }

    truth_rows = []
    global_event_id = 0

    for g in range(1, int(args.n_groups) + 1):
        group_name = f"{args.group_prefix}_{g:02d}"
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for label in LABEL_ORDER:
            label_dir = group_dir / folder_names[label]
            label_dir.mkdir(parents=True, exist_ok=True)
            spectra_for_merged: List[np.ndarray] = []
            names_for_merged: List[str] = []

            for i in range(1, counts[label] + 1):
                y, truth = noisy_spectrum(
                    x=x,
                    r6g_template=r6g_template,
                    mpy_template=mpy_template,
                    label=label,
                    rng=rng,
                    signal_scale=float(args.signal_scale),
                    noise_std=float(args.noise_std),
                    none_noise_std=float(args.none_noise_std),
                    baseline_strength=float(args.baseline_strength),
                    amp_cv=float(args.amplitude_cv),
                )
                filename = f"saved_col_{i:06d}.csv"
                file_path = label_dir / filename
                if not args.no_single_files:
                    write_single_csv(file_path, x, y)
                spectra_for_merged.append(y)
                names_for_merged.append(filename)

                truth_rows.append({
                    "event_id": global_event_id,
                    "group": group_name,
                    "label": label,
                    "terminal_dir": str(label_dir),
                    "source_file": str(file_path),
                    "source_column": filename,
                    **truth,
                })
                global_event_id += 1

            if not args.no_merged:
                write_merged_csv(label_dir / "merged.csv", x, spectra_for_merged, names_for_merged)

    template_df = pd.DataFrame({
        "Raman shift (cm-1)": x,
        "R6G_template": r6g_template,
        "4-MPY_template": mpy_template,
    })
    template_df.to_csv(out_dir / "synthetic_templates.csv", index=False)

    truth_df = pd.DataFrame(truth_rows)
    truth_df.to_csv(out_dir / "synthetic_ground_truth_metadata.csv", index=False)

    summary = {
        "description": "Synthetic R6G/4-MPY SERS dataset for MPCA validation",
        "out_dir": str(out_dir),
        "n_groups": int(args.n_groups),
        "counts_per_group": counts,
        "total_events": int(len(truth_rows)),
        "x_start": float(args.x_start),
        "x_end": float(args.x_end),
        "n_points": int(args.n_points),
        "r6g_peaks": r6g_peaks,
        "mpy_peaks": mpy_peaks,
        "signal_scale": float(args.signal_scale),
        "noise_std": float(args.noise_std),
        "none_noise_std": float(args.none_noise_std),
        "baseline_strength": float(args.baseline_strength),
        "amplitude_cv": float(args.amplitude_cv),
        "seed": int(args.seed),
        "writes_merged_csv": not args.no_merged,
        "writes_single_files": not args.no_single_files,
        "recommended_mpca_command": (
            f"python mpca_sers_analysis_v3.py --data-dir {out_dir} "
            f"--out-dir results/synthetic_mpca_test --x-range 500 1800 "
            f"--n-points 1301 --exclude-none-for-pca --per-group --qc"
        ),
    }
    with open(out_dir / "synthetic_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Synthetic dataset saved to: {out_dir}")
    print(f"[OK] Total events: {len(truth_rows)}")
    print(f"[OK] Ground truth: {out_dir / 'synthetic_ground_truth_metadata.csv'}")
    print(f"[OK] Templates: {out_dir / 'synthetic_templates.csv'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic R6G/4-MPY SERS spectra with Lorentzian peaks and Gaussian noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output root directory for the synthetic dataset.")
    p.add_argument("--overwrite", action="store_true", help="Delete and recreate --out-dir if it already exists.")
    p.add_argument("--seed", type=int, default=202606, help="Random seed.")

    p.add_argument("--n-groups", type=int, default=1, help="Number of synthetic experimental groups.")
    p.add_argument("--group-prefix", default="synthetic_group", help="Prefix for group folders.")
    p.add_argument("--n-r6g", type=int, default=44, help="R6G-only spectra per group.")
    p.add_argument("--n-mpy", type=int, default=51, help="4-MPY-only spectra per group.")
    p.add_argument("--n-mix", type=int, default=11, help="Mix spectra per group.")
    p.add_argument("--n-none", type=int, default=300, help="None/no-signal spectra per group.")

    p.add_argument("--x-start", type=float, default=300.0, help="Minimum Raman shift in generated raw CSV files.")
    p.add_argument("--x-end", type=float, default=2000.0, help="Maximum Raman shift in generated raw CSV files.")
    p.add_argument("--n-points", type=int, default=1701, help="Number of raw x-axis points.")

    p.add_argument("--signal-scale", type=float, default=1000.0, help="Overall signal scale for analyte spectra.")
    p.add_argument("--noise-std", type=float, default=20.0, help="Gaussian noise standard deviation for R6G/4-MPY/Mix spectra.")
    p.add_argument("--none-noise-std", type=float, default=20.0, help="Gaussian noise standard deviation for None spectra.")
    p.add_argument("--baseline-strength", type=float, default=0.0, help="Weak random polynomial baseline strength. Use 0 for ideal data.")
    p.add_argument("--amplitude-cv", type=float, default=0.25, help="Lognormal amplitude variation; larger means stronger event-to-event variability.")

    p.add_argument("--r6g-peaks", default=None, help="Comma-separated synthetic R6G peak centers.")
    p.add_argument("--r6g-heights", default=None, help="Comma-separated synthetic R6G relative peak heights.")
    p.add_argument("--r6g-widths", default=None, help="Comma-separated synthetic R6G FWHM values.")
    p.add_argument("--mpy-peaks", default=None, help="Comma-separated synthetic 4-MPY peak centers.")
    p.add_argument("--mpy-heights", default=None, help="Comma-separated synthetic 4-MPY relative peak heights.")
    p.add_argument("--mpy-widths", default=None, help="Comma-separated synthetic 4-MPY FWHM values.")

    p.add_argument("--no-merged", action="store_true", help="Do not write merged.csv files.")
    p.add_argument("--no-single-files", action="store_true", help="Do not write individual saved_col_*.csv files.")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_points < 3:
        parser.error("--n-points must be >= 3")
    if args.x_start >= args.x_end:
        parser.error("--x-start must be smaller than --x-end")
    for name in ["n_groups", "n_r6g", "n_mpy", "n_mix", "n_none"]:
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    try:
        generate_dataset(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
