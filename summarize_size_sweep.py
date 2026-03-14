#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^(?P<variant>baseline|without_influencer|with_influencer)"
    r"_size(?P<size>\d+)"
    r"_seed(?P<seed>-?\d+)"
    r"_L(?P<layers>\d+)"
    r"_(?P<similarity>.+)$"
)

FINAL_METRICS = ["moves", "tau", "entropy", "gini", "drift"]
CLUSTER_METRICS = [
    "total_clusters",
    "num_clusters_red",
    "num_clusters_green",
    "num_clusters_blue",
    "largest_red",
    "largest_green",
    "largest_blue",
    "total_red",
    "total_green",
    "total_blue",
    "frac_largest_red",
    "frac_largest_green",
    "frac_largest_blue",
]


def parse_run_tag(stem: str) -> dict:
    m = RUN_RE.match(stem)
    if not m:
        raise ValueError(f"Could not parse run tag from: {stem}")
    out = m.groupdict()
    out["size"] = int(out["size"])
    out["seed"] = int(out["seed"])
    out["layers"] = int(out["layers"])
    return out


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ci95(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(series)
    if n <= 1:
        return 0.0
    return 1.96 * float(series.std(ddof=1)) / math.sqrt(n)


def first_zero_step(series: pd.Series, steps: pd.Series) -> float:
    mask = pd.to_numeric(series, errors="coerce").fillna(np.nan).eq(0)
    if not mask.any():
        return np.nan
    return float(pd.to_numeric(steps, errors="coerce").loc[mask].iloc[0])


def stable_step(series: pd.Series, steps: pd.Series, tol: float = 1e-9) -> float:
    series = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    steps = pd.to_numeric(steps, errors="coerce").reset_index(drop=True)
    if len(series) < 2:
        return np.nan
    diffs = series.diff().abs()
    stable = diffs.le(tol)
    stable.iloc[0] = False
    if stable.any():
        idx = stable[stable].index[0]
        return float(steps.iloc[idx])
    return np.nan


def discrete_auc(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float(vals.sum())


def last_minus_prev(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return np.nan
    return float(vals[-1] - vals[-2])


def read_metrics_files(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        meta = parse_run_tag(csv_path.stem)
        df = pd.read_csv(csv_path)
        for k, v in meta.items():
            df[k] = v
        df["run_tag"] = csv_path.stem
        df["metrics_file"] = str(csv_path)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No metrics CSV files found in {metrics_dir}")

    out = pd.concat(rows, ignore_index=True)
    out = out[
        [
            "variant",
            "size",
            "seed",
            "layers",
            "similarity",
            "run_tag",
            "step",
            "moves",
            "tau",
            "entropy",
            "gini",
            "drift",
            "metrics_file",
        ]
    ].sort_values(["variant", "size", "seed", "layers", "step"]).reset_index(drop=True)
    return out


def build_final_from_steps(master_steps: pd.DataFrame) -> pd.DataFrame:
    idx = master_steps.groupby("run_tag")["step"].idxmax()
    final_df = master_steps.loc[idx].copy()
    final_df = final_df.rename(
        columns={
            "step": "final_step",
            "moves": "final_moves",
            "tau": "final_tau",
            "entropy": "final_entropy",
            "gini": "final_gini",
            "drift": "final_drift",
        }
    )
    keep_cols = [
        "variant",
        "size",
        "seed",
        "layers",
        "similarity",
        "run_tag",
        "final_step",
        "final_moves",
        "final_tau",
        "final_entropy",
        "final_gini",
        "final_drift",
    ]
    return final_df[keep_cols].sort_values(
        ["variant", "size", "seed", "layers"]
    ).reset_index(drop=True)


def read_existing_final_table(base_dir: Path) -> pd.DataFrame | None:
    path = base_dir / "table_sizes_20_to_200.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["run_tag"] = (
        df["variant"]
        + "_size"
        + df["size"].astype(int).astype(str)
        + "_seed"
        + df["seed"].astype(int).astype(str)
        + "_L"
        + df["layers"].astype(int).astype(str)
        + "_"
        + df["similarity"].astype(str)
    )
    df = df.rename(
        columns={
            "step": "final_step",
            "moves": "final_moves",
            "tau": "final_tau",
            "entropy": "final_entropy",
            "gini": "final_gini",
            "drift": "final_drift",
        }
    )
    keep_cols = [
        "variant",
        "size",
        "seed",
        "layers",
        "similarity",
        "run_tag",
        "final_step",
        "final_moves",
        "final_tau",
        "final_entropy",
        "final_gini",
        "final_drift",
    ]
    return df[keep_cols].sort_values(["variant", "size", "seed", "layers"]).reset_index(drop=True)


def read_final_clusters(base_dir: Path) -> pd.DataFrame | None:
    path = base_dir / "final_clusters.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["run_tag"] = (
        df["variant"]
        + "_size"
        + df["size"].astype(int).astype(str)
        + "_seed"
        + df["seed"].astype(int).astype(str)
        + "_L"
        + df["layers"].astype(int).astype(str)
        + "_"
        + df["similarity"].astype(str)
    )
    ordered = [
        "variant",
        "size",
        "seed",
        "layers",
        "similarity",
        "run_tag",
        *CLUSTER_METRICS,
    ]
    return df[ordered].sort_values(["variant", "size", "seed", "layers"]).reset_index(drop=True)


def read_cluster_distributions(base_dir: Path) -> pd.DataFrame | None:
    path = base_dir / "final_cluster_distributions.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["cluster_sizes"] = df["cluster_sizes"].fillna("")
    long_rows = []

    for _, row in df.iterrows():
        sizes = [int(x) for x in str(row["cluster_sizes"]).split(";") if str(x).strip()]
        if not sizes:
            long_rows.append(
                {
                    "variant": row["variant"],
                    "size": int(row["size"]),
                    "seed": int(row["seed"]),
                    "layers": int(row["layers"]),
                    "similarity": row["similarity"],
                    "species": row["species"],
                    "num_clusters": int(row["num_clusters"]),
                    "cluster_rank": np.nan,
                    "cluster_size": np.nan,
                }
            )
            continue

        for rank, size in enumerate(sorted(sizes, reverse=True), start=1):
            long_rows.append(
                {
                    "variant": row["variant"],
                    "size": int(row["size"]),
                    "seed": int(row["seed"]),
                    "layers": int(row["layers"]),
                    "similarity": row["similarity"],
                    "species": row["species"],
                    "num_clusters": int(row["num_clusters"]),
                    "cluster_rank": rank,
                    "cluster_size": size,
                }
            )

    return pd.DataFrame(long_rows)


def compute_per_run_dynamics(master_steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_tag, g in master_steps.groupby("run_tag", sort=False):
        g = g.sort_values("step")
        rows.append(
            {
                "variant": g["variant"].iloc[0],
                "size": int(g["size"].iloc[0]),
                "seed": int(g["seed"].iloc[0]),
                "layers": int(g["layers"].iloc[0]),
                "similarity": g["similarity"].iloc[0],
                "run_tag": run_tag,
                "n_steps": int(g["step"].max()),
                "auc_moves": discrete_auc(g["moves"]),
                "auc_tau": discrete_auc(g["tau"]),
                "auc_entropy": discrete_auc(g["entropy"]),
                "auc_gini": discrete_auc(g["gini"]),
                "auc_drift": discrete_auc(g["drift"]),
                "first_zero_moves_step": first_zero_step(g["moves"], g["step"]),
                "stable_moves_step": stable_step(g["moves"], g["step"], tol=0.0),
                "stable_tau_step": stable_step(g["tau"], g["step"], tol=1e-12),
                "stable_entropy_step": stable_step(g["entropy"], g["step"], tol=1e-12),
                "stable_gini_step": stable_step(g["gini"], g["step"], tol=1e-12),
                "stable_drift_step": stable_step(g["drift"], g["step"], tol=1e-12),
                "delta_moves_last": last_minus_prev(g["moves"]),
                "delta_tau_last": last_minus_prev(g["tau"]),
                "delta_entropy_last": last_minus_prev(g["entropy"]),
                "delta_gini_last": last_minus_prev(g["gini"]),
                "delta_drift_last": last_minus_prev(g["drift"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["variant", "size", "seed", "layers"]).reset_index(drop=True)


def aggregate_numeric(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    out_frames = []
    for col in value_cols:
        agg = (
            df.groupby(group_cols, dropna=False)[col]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        ci = (
            df.groupby(group_cols, dropna=False)[col]
            .apply(ci95)
            .reset_index(name=f"{col}_ci95")
        )
        agg = agg.merge(ci, on=group_cols, how="left")
        agg = agg.rename(
            columns={
                "count": f"{col}_n",
                "mean": f"{col}_mean",
                "std": f"{col}_std",
                "median": f"{col}_median",
                "min": f"{col}_min",
                "max": f"{col}_max",
            }
        )
        out_frames.append(agg)

    merged = out_frames[0]
    for extra in out_frames[1:]:
        merged = merged.merge(extra, on=group_cols, how="outer")

    return merged.sort_values(group_cols).reset_index(drop=True)


def build_master_runs(
    final_metrics: pd.DataFrame,
    cluster_df: pd.DataFrame | None,
    per_run_dynamics: pd.DataFrame,
) -> pd.DataFrame:
    master_runs = final_metrics.merge(
        per_run_dynamics,
        on=["variant", "size", "seed", "layers", "similarity", "run_tag"],
        how="left",
    )
    if cluster_df is not None:
        master_runs = master_runs.merge(
            cluster_df,
            on=["variant", "size", "seed", "layers", "similarity", "run_tag"],
            how="left",
        )
    return master_runs.sort_values(["variant", "size", "seed", "layers"]).reset_index(drop=True)


def add_variant_layer_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["variant_layers"] = np.where(
        out["variant"].eq("baseline"),
        "baseline_L1",
        out["variant"] + "_L" + out["layers"].astype(int).astype(str),
    )
    return out


def plot_scaling(summary_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    summary_df = add_variant_layer_label(summary_df)

    metrics = [
        ("final_entropy_mean", "Final entropy"),
        ("final_gini_mean", "Final gini"),
        ("final_drift_mean", "Final drift"),
        ("final_moves_mean", "Final moves"),
        ("total_clusters_mean", "Total clusters"),
        ("frac_largest_red_mean", "Largest cluster fraction (red)"),
        ("frac_largest_green_mean", "Largest cluster fraction (green)"),
        ("frac_largest_blue_mean", "Largest cluster fraction (blue)"),
    ]

    for col, ylabel in metrics:
        if col not in summary_df.columns:
            continue

        plt.figure(figsize=(8, 5))
        for name, g in summary_df.groupby("variant_layers", sort=True):
            g = g.sort_values("size")
            plt.plot(g["size"], g[col], marker="o", label=name)
            ci_col = col.replace("_mean", "_ci95")
            if ci_col in g.columns:
                lower = g[col] - g[ci_col]
                upper = g[col] + g[ci_col]
                plt.fill_between(g["size"], lower, upper, alpha=0.15)

        plt.xlabel("Grid size")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs grid size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_vs_size.png", dpi=200)
        plt.close()


def plot_dynamics(step_summary: pd.DataFrame, out_dir: Path, rep_sizes: Iterable[int]) -> None:
    ensure_dir(out_dir)
    step_summary = add_variant_layer_label(step_summary)

    metrics = [
        ("moves_mean", "Moves"),
        ("tau_mean", "Tau"),
        ("entropy_mean", "Entropy"),
        ("gini_mean", "Gini"),
        ("drift_mean", "Drift"),
    ]

    for size in rep_sizes:
        gsize = step_summary[step_summary["size"] == size].copy()
        if gsize.empty:
            continue

        for col, ylabel in metrics:
            if col not in gsize.columns:
                continue

            plt.figure(figsize=(8, 5))
            for name, g in gsize.groupby("variant_layers", sort=True):
                g = g.sort_values("step")
                plt.plot(g["step"], g[col], marker="o", label=name)
                ci_col = col.replace("_mean", "_ci95")
                if ci_col in g.columns:
                    lower = g[col] - g[ci_col]
                    upper = g[col] + g[ci_col]
                    plt.fill_between(g["step"], lower, upper, alpha=0.15)

            plt.xlabel("Step")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} over time (size={size})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"size_{size}_{col}_over_time.png", dpi=200)
            plt.close()


def plot_cluster_distributions(cluster_long: pd.DataFrame | None, out_dir: Path, rep_sizes: Iterable[int]) -> None:
    if cluster_long is None or cluster_long.empty:
        return

    ensure_dir(out_dir)
    tmp = cluster_long.dropna(subset=["cluster_size"]).copy()
    if tmp.empty:
        return

    tmp["variant_layers"] = np.where(
        tmp["variant"].eq("baseline"),
        "baseline_L1",
        tmp["variant"] + "_L" + tmp["layers"].astype(int).astype(str),
    )

    for size in rep_sizes:
        df_size = tmp[tmp["size"] == size].copy()
        if df_size.empty:
            continue

        for species in sorted(df_size["species"].dropna().unique()):
            ds = df_size[df_size["species"] == species]

            plt.figure(figsize=(8, 5))
            for name, g in ds.groupby("variant_layers", sort=True):
                vals = g["cluster_size"].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                vals = np.sort(vals)
                ccdf = 1.0 - np.arange(1, len(vals) + 1) / len(vals)
                plt.plot(vals, ccdf, marker=".", linestyle="-", label=name)

            plt.xlabel("Cluster size")
            plt.ylabel("CCDF")
            plt.title(f"Cluster size tail (size={size}, species={species})")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"cluster_ccdf_size_{size}_{species}.png", dpi=200)
            plt.close()


def choose_rep_sizes(df: pd.DataFrame) -> list[int]:
    sizes = sorted(pd.to_numeric(df["size"], errors="coerce").dropna().astype(int).unique().tolist())
    if not sizes:
        return []

    preferred = [40, 100, 200]
    picked = [s for s in preferred if s in sizes]
    if len(picked) >= 3:
        return picked

    extra = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
    out = []
    for s in picked + extra:
        if s not in out:
            out.append(s)
    return out[:3]


def save_dataframes(
    analysis_dir: Path,
    master_steps: pd.DataFrame,
    master_runs: pd.DataFrame,
    step_summary_vsl: pd.DataFrame,
    final_summary_vsl: pd.DataFrame,
    final_summary_vs: pd.DataFrame,
    cluster_long: pd.DataFrame | None,
) -> None:
    ensure_dir(analysis_dir)
    master_steps.to_csv(analysis_dir / "master_steps.csv", index=False)
    master_runs.to_csv(analysis_dir / "master_runs.csv", index=False)
    step_summary_vsl.to_csv(analysis_dir / "step_summary_by_variant_size_layers.csv", index=False)
    final_summary_vsl.to_csv(analysis_dir / "final_summary_by_variant_size_layers.csv", index=False)
    final_summary_vs.to_csv(analysis_dir / "final_summary_by_variant_size.csv", index=False)
    if cluster_long is not None:
        cluster_long.to_csv(analysis_dir / "final_cluster_distributions_long.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize outputs/size_sweep into thesis-ready tables and plots.")
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/size_sweep"))
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    analysis_dir = args.analysis_dir or (base_dir / "analysis")
    plots_dir = args.plots_dir or (base_dir / "plots")

    metrics_dir = base_dir / "metrics"
    master_steps = read_metrics_files(metrics_dir)

    final_from_steps = build_final_from_steps(master_steps)
    existing_final = read_existing_final_table(base_dir)
    final_metrics = existing_final if existing_final is not None else final_from_steps

    cluster_df = read_final_clusters(base_dir)
    cluster_long = read_cluster_distributions(base_dir)
    per_run_dynamics = compute_per_run_dynamics(master_steps)
    master_runs = build_master_runs(final_metrics, cluster_df, per_run_dynamics)

    final_value_cols = [
        "final_moves",
        "final_tau",
        "final_entropy",
        "final_gini",
        "final_drift",
        "auc_moves",
        "auc_tau",
        "auc_entropy",
        "auc_gini",
        "auc_drift",
        "first_zero_moves_step",
        "stable_moves_step",
        "stable_tau_step",
        "stable_entropy_step",
        "stable_gini_step",
        "stable_drift_step",
        "delta_moves_last",
        "delta_tau_last",
        "delta_entropy_last",
        "delta_gini_last",
        "delta_drift_last",
    ]
    final_value_cols = [c for c in final_value_cols if c in master_runs.columns]

    if cluster_df is not None:
        final_value_cols += [c for c in CLUSTER_METRICS if c in master_runs.columns]

    step_summary_vsl = aggregate_numeric(
        master_steps,
        ["variant", "size", "layers", "step"],
        FINAL_METRICS,
    )

    final_summary_vsl = aggregate_numeric(
        master_runs,
        ["variant", "size", "layers"],
        final_value_cols,
    )

    final_summary_vs = aggregate_numeric(
        master_runs,
        ["variant", "size"],
        final_value_cols,
    )

    save_dataframes(
        analysis_dir,
        master_steps=master_steps,
        master_runs=master_runs,
        step_summary_vsl=step_summary_vsl,
        final_summary_vsl=final_summary_vsl,
        final_summary_vs=final_summary_vs,
        cluster_long=cluster_long,
    )

    rep_sizes = choose_rep_sizes(master_runs)
    plot_scaling(final_summary_vsl, ensure_dir(plots_dir / "scaling"))
    plot_dynamics(step_summary_vsl, ensure_dir(plots_dir / "dynamics"), rep_sizes)
    plot_cluster_distributions(cluster_long, ensure_dir(plots_dir / "clusters"), rep_sizes)

    compact_cols = [
        "variant",
        "size",
        "layers",
        "final_entropy_mean",
        "final_entropy_std",
        "final_gini_mean",
        "final_gini_std",
        "final_drift_mean",
        "final_drift_std",
        "final_moves_mean",
        "final_moves_std",
        "total_clusters_mean",
        "total_clusters_std",
        "frac_largest_red_mean",
        "frac_largest_green_mean",
        "frac_largest_blue_mean",
    ]
    compact_cols = [c for c in compact_cols if c in final_summary_vsl.columns]
    final_summary_vsl[compact_cols].to_csv(analysis_dir / "table_main_results_compact.csv", index=False)

    print(f"[done] Wrote analysis tables to: {analysis_dir}")
    print(f"[done] Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()