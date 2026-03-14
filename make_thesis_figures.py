#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def add_condition_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["layers"] = pd.to_numeric(out["layers"], errors="coerce").astype("Int64")
    out["condition"] = np.where(
        out["variant"].eq("baseline"),
        "baseline",
        out["variant"] + "_L" + out["layers"].astype(str),
    )
    return out


def ordered_conditions(df: pd.DataFrame) -> list[str]:
    desired = [
        "baseline",
        "without_influencer_L1",
        "without_influencer_L2",
        "without_influencer_L3",
        "with_influencer_L1",
        "with_influencer_L2",
        "with_influencer_L3",
    ]
    present = [c for c in desired if c in set(df["condition"].dropna())]
    extra = sorted([c for c in set(df["condition"].dropna()) if c not in set(present)])
    return present + extra


def representative_sizes(sizes: list[int]) -> list[int]:
    sizes = sorted(set(int(s) for s in sizes))
    if not sizes:
        return []
    preferred = [40, 100, 140, 200]
    picked = [s for s in preferred if s in sizes]
    if len(picked) >= 3:
        return picked[:3]
    fallback = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
    out = []
    for s in picked + fallback:
        if s not in out:
            out.append(s)
    return out[:3]


def add_normalized_columns(
    master_runs: pd.DataFrame,
    final_summary: pd.DataFrame,
    master_steps: pd.DataFrame,
    step_summary: pd.DataFrame,
) -> None:
    if {"total_red", "total_green", "total_blue"}.issubset(master_runs.columns):
        master_runs["occupied_final"] = (
            master_runs["total_red"] + master_runs["total_green"] + master_runs["total_blue"]
        )
        master_runs["final_move_rate"] = master_runs["final_moves"] / master_runs["occupied_final"].replace(0, np.nan)
        master_runs["mean_cluster_size"] = master_runs["occupied_final"] / master_runs["total_clusters"].replace(0, np.nan)

    if {"total_red_mean", "total_green_mean", "total_blue_mean"}.issubset(final_summary.columns):
        final_summary["occupied_final_mean"] = (
            final_summary["total_red_mean"]
            + final_summary["total_green_mean"]
            + final_summary["total_blue_mean"]
        )
        final_summary["final_move_rate_mean"] = (
            final_summary["final_moves_mean"] / final_summary["occupied_final_mean"].replace(0, np.nan)
        )
        final_summary["mean_cluster_size_mean"] = (
            final_summary["occupied_final_mean"] / final_summary["total_clusters_mean"].replace(0, np.nan)
        )

    if {"size"}.issubset(master_steps.columns):
        master_steps["grid_cells"] = master_steps["size"] ** 2

    if {"size", "moves_mean"}.issubset(step_summary.columns):
        step_summary["occupied_expected"] = 0.78 * (step_summary["size"] ** 2)
        step_summary["moves_rate_mean"] = (
            step_summary["moves_mean"] / step_summary["occupied_expected"].replace(0, np.nan)
        )


def plot_metric_vs_size(
    df: pd.DataFrame,
    y: str,
    y_ci: str | None,
    ylabel: str,
    title: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    plt.figure(figsize=(8.5, 5.5))
    for cond in ordered_conditions(df):
        g = df[df["condition"] == cond].sort_values("size")
        if g.empty or y not in g.columns:
            continue
        plt.plot(g["size"], g[y], marker="o", linewidth=2, label=cond)
        if y_ci and y_ci in g.columns:
            lower = g[y] - g[y_ci]
            upper = g[y] + g[y_ci]
            plt.fill_between(g["size"], lower, upper, alpha=0.15)
    plt.xlabel("Grid size L")
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_over_time(
    df: pd.DataFrame,
    size: int,
    y: str,
    y_ci: str | None,
    ylabel: str,
    title: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    subset = df[df["size"] == size].copy()
    if subset.empty:
        return
    plt.figure(figsize=(8.5, 5.5))
    for cond in ordered_conditions(subset):
        g = subset[subset["condition"] == cond].sort_values("step")
        if g.empty or y not in g.columns:
            continue
        plt.plot(g["step"], g[y], marker="o", linewidth=2, label=cond)
        if y_ci and y_ci in g.columns:
            lower = g[y] - g[y_ci]
            upper = g[y] + g[y_ci]
            plt.fill_between(g["step"], lower, upper, alpha=0.15)
    plt.xlabel("Simulation step")
    plt.ylabel(ylabel)
    plt.title(f"{title} (L={size})")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_seed_boxplots(
    master_runs: pd.DataFrame,
    metric: str,
    ylabel: str,
    sizes: list[int],
    out_path: Path,
) -> None:
    subset = master_runs[master_runs["size"].isin(sizes)].copy()
    if subset.empty or metric not in subset.columns:
        return

    subset["group"] = subset["condition"] + "\nL=" + subset["size"].astype(str)
    order = []
    for s in sizes:
        ds = subset[subset["size"] == s]
        for c in ordered_conditions(ds):
            name = f"{c}\nL={s}"
            if name in set(ds["group"]):
                order.append(name)

    data = [subset.loc[subset["group"] == g, metric].dropna().to_numpy() for g in order]

    plt.figure(figsize=(max(12, 0.9 * len(order)), 5.5))
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(f"Across-seed distribution of {ylabel.lower()}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scatter(
    master_runs: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 5.5))
    for cond in ordered_conditions(master_runs):
        g = master_runs[master_runs["condition"] == cond]
        if g.empty or x not in g.columns or y not in g.columns:
            continue
        plt.scatter(g[x], g[y], alpha=0.75, label=cond)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def learned_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["variant"] != "baseline"].copy()


def plot_heatmap_for_variant(
    df: pd.DataFrame,
    variant: str,
    value_col: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    subset = df[df["variant"] == variant].copy()
    if subset.empty or value_col not in subset.columns:
        return
    pivot = subset.pivot(index="layers", columns="size", values=value_col).sort_index().sort_index(axis=1)
    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar(im, label=value_col.replace("_", " "))
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Grid size L")
    plt.ylabel("Layers K")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cluster_ccdf(
    cluster_long: pd.DataFrame,
    size: int,
    species: str,
    out_path: Path,
) -> None:
    subset = cluster_long[(cluster_long["size"] == size) & (cluster_long["species"] == species)].copy()
    subset = subset.dropna(subset=["cluster_size"])
    if subset.empty:
        return

    plt.figure(figsize=(7.5, 5.5))
    for cond in ordered_conditions(subset):
        g = subset[subset["condition"] == cond]
        vals = g["cluster_size"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        vals = np.sort(vals)
        ccdf = 1.0 - np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, ccdf, marker=".", linestyle="-", label=cond)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cluster size")
    plt.ylabel("CCDF")
    plt.title(f"Cluster-size tail, species={species}, L={size}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_largest_cluster_species(df: pd.DataFrame, out_path: Path) -> None:
    needed = ["frac_largest_red_mean", "frac_largest_green_mean", "frac_largest_blue_mean"]
    if not set(needed).issubset(df.columns):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    species_cols = [
        ("frac_largest_red_mean", "Red"),
        ("frac_largest_green_mean", "Green"),
        ("frac_largest_blue_mean", "Blue"),
    ]
    for ax, (col, title) in zip(axes, species_cols):
        for cond in ordered_conditions(df):
            g = df[df["condition"] == cond].sort_values("size")
            if g.empty:
                continue
            ax.plot(g["size"], g[col], marker="o", linewidth=2, label=cond)
        ax.set_title(title)
        ax.set_xlabel("Grid size L")
        ax.set_ylabel("Largest cluster fraction")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Largest connected-component fraction by species", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auc_vs_size(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    plot_metric_vs_size(
        df=df,
        y=f"{value_col}_mean",
        y_ci=f"{value_col}_ci95" if f"{value_col}_ci95" in df.columns else None,
        ylabel=ylabel,
        title=f"{ylabel} vs grid size",
        out_path=out_path,
        logy=logy,
    )


def create_all_figures(data_dir: Path, out_dir: Path) -> None:
    master_runs = read_csv_required(data_dir / "master_runs.csv")
    master_steps = read_csv_required(data_dir / "master_steps.csv")
    final_summary = read_csv_required(data_dir / "final_summary_by_variant_size_layers.csv")
    step_summary = read_csv_required(data_dir / "step_summary_by_variant_size_layers.csv")
    cluster_path = data_dir / "final_cluster_distributions_long.csv"
    cluster_long = pd.read_csv(cluster_path) if cluster_path.exists() else None

    master_runs = add_condition_label(master_runs)
    master_steps = add_condition_label(master_steps)
    final_summary = add_condition_label(final_summary)
    step_summary = add_condition_label(step_summary)
    if cluster_long is not None:
        cluster_long = add_condition_label(cluster_long)

    add_normalized_columns(master_runs, final_summary, master_steps, step_summary)

    sizes = sorted(pd.to_numeric(master_runs["size"], errors="coerce").dropna().astype(int).unique().tolist())
    rep_sizes = representative_sizes(sizes)

    core_dir = ensure_dir(out_dir / "01_core_scaling")
    dyn_dir = ensure_dir(out_dir / "02_dynamics")
    cluster_dir = ensure_dir(out_dir / "03_clusters")
    box_dir = ensure_dir(out_dir / "04_seed_distributions")
    heat_dir = ensure_dir(out_dir / "05_heatmaps")
    scatter_dir = ensure_dir(out_dir / "06_relationships")

    plot_metric_vs_size(
        final_summary,
        "final_entropy_mean",
        "final_entropy_ci95",
        "Final entropy",
        "Final entropy vs grid size",
        core_dir / "01_final_entropy_vs_size.png",
    )
    plot_metric_vs_size(
        final_summary,
        "final_gini_mean",
        "final_gini_ci95",
        "Final Gini impurity",
        "Final Gini impurity vs grid size",
        core_dir / "02_final_gini_vs_size.png",
    )
    plot_metric_vs_size(
        final_summary,
        "final_drift_mean",
        "final_drift_ci95",
        "Final drift",
        "Final drift vs grid size",
        core_dir / "03_final_drift_vs_size.png",
    )
    plot_metric_vs_size(
        final_summary,
        "final_moves_mean",
        "final_moves_ci95",
        "Final moves",
        "Final moves vs grid size",
        core_dir / "04_final_moves_vs_size.png",
        logy=True,
    )

    if "final_move_rate_mean" in final_summary.columns:
        plot_metric_vs_size(
            final_summary,
            "final_move_rate_mean",
            None,
            "Final move rate",
            "Final move rate vs grid size",
            core_dir / "05_final_move_rate_vs_size.png",
            logy=True,
        )

    plot_metric_vs_size(
        final_summary,
        "total_clusters_mean",
        "total_clusters_ci95",
        "Total clusters",
        "Final cluster count vs grid size",
        core_dir / "06_total_clusters_vs_size.png",
        logy=True,
    )

    if "mean_cluster_size_mean" in final_summary.columns:
        plot_metric_vs_size(
            final_summary,
            "mean_cluster_size_mean",
            None,
            "Mean cluster size",
            "Mean cluster size vs grid size",
            core_dir / "07_mean_cluster_size_vs_size.png",
            logy=True,
        )

    plot_largest_cluster_species(
        final_summary,
        core_dir / "08_largest_cluster_fraction_by_species.png",
    )

    if "auc_moves_mean" in final_summary.columns:
        plot_auc_vs_size(
            final_summary,
            "auc_moves",
            "AUC of moves",
            core_dir / "09_auc_moves_vs_size.png",
            logy=True,
        )

    if "auc_drift_mean" in final_summary.columns:
        plot_auc_vs_size(
            final_summary,
            "auc_drift",
            "AUC of drift",
            core_dir / "10_auc_drift_vs_size.png",
        )

    if "first_zero_moves_step_mean" in final_summary.columns:
        plot_metric_vs_size(
            final_summary,
            "first_zero_moves_step_mean",
            "first_zero_moves_step_ci95",
            "First zero-move step",
            "First zero-move step vs grid size",
            core_dir / "11_first_zero_move_step_vs_size.png",
        )

    for size in rep_sizes:
        plot_metric_over_time(
            step_summary,
            size,
            "moves_mean",
            "moves_ci95",
            "Moves",
            "Movement dynamics",
            dyn_dir / f"L{size:03d}_01_moves_over_time.png",
            logy=True,
        )
        if "moves_rate_mean" in step_summary.columns:
            plot_metric_over_time(
                step_summary,
                size,
                "moves_rate_mean",
                None,
                "Move rate",
                "Normalized movement dynamics",
                dyn_dir / f"L{size:03d}_02_move_rate_over_time.png",
                logy=True,
            )
        plot_metric_over_time(
            step_summary,
            size,
            "tau_mean",
            "tau_ci95",
            "Tau",
            "Adaptive threshold dynamics",
            dyn_dir / f"L{size:03d}_03_tau_over_time.png",
        )
        plot_metric_over_time(
            step_summary,
            size,
            "entropy_mean",
            "entropy_ci95",
            "Entropy",
            "Entropy dynamics",
            dyn_dir / f"L{size:03d}_04_entropy_over_time.png",
        )
        plot_metric_over_time(
            step_summary,
            size,
            "gini_mean",
            "gini_ci95",
            "Gini impurity",
            "Gini dynamics",
            dyn_dir / f"L{size:03d}_05_gini_over_time.png",
        )
        plot_metric_over_time(
            step_summary,
            size,
            "drift_mean",
            "drift_ci95",
            "Drift",
            "Drift dynamics",
            dyn_dir / f"L{size:03d}_06_drift_over_time.png",
        )

    if cluster_long is not None and not cluster_long.empty:
        max_size = max(sizes)
        med_size = rep_sizes[1] if len(rep_sizes) >= 2 else max_size
        for s in sorted(set([med_size, max_size])):
            for sp in ["red", "green", "blue"]:
                plot_cluster_ccdf(
                    cluster_long,
                    s,
                    sp,
                    cluster_dir / f"cluster_ccdf_L{s}_{sp}.png",
                )

    selected_sizes = rep_sizes
    plot_seed_boxplots(
        master_runs,
        "final_entropy",
        "Final entropy",
        selected_sizes,
        box_dir / "01_boxplot_final_entropy.png",
    )
    plot_seed_boxplots(
        master_runs,
        "final_moves",
        "Final moves",
        selected_sizes,
        box_dir / "02_boxplot_final_moves.png",
    )
    plot_seed_boxplots(
        master_runs,
        "final_drift",
        "Final drift",
        selected_sizes,
        box_dir / "03_boxplot_final_drift.png",
    )
    plot_seed_boxplots(
        master_runs,
        "total_clusters",
        "Total clusters",
        selected_sizes,
        box_dir / "04_boxplot_total_clusters.png",
    )

    learned = learned_only(final_summary)
    for variant in ["without_influencer", "with_influencer"]:
        plot_heatmap_for_variant(
            learned,
            variant,
            "final_entropy_mean",
            f"{variant}: final entropy",
            heat_dir / f"{variant}_final_entropy_heatmap.png",
        )
        plot_heatmap_for_variant(
            learned,
            variant,
            "final_moves_mean",
            f"{variant}: final moves",
            heat_dir / f"{variant}_final_moves_heatmap.png",
            cmap="magma",
        )
        if "final_drift_mean" in learned.columns:
            plot_heatmap_for_variant(
                learned,
                variant,
                "final_drift_mean",
                f"{variant}: final drift",
                heat_dir / f"{variant}_final_drift_heatmap.png",
            )

    plot_scatter(
        master_runs,
        "final_entropy",
        "final_moves",
        "Final entropy",
        "Final moves",
        "Entropy-mobility relationship",
        scatter_dir / "01_entropy_vs_final_moves.png",
    )
    plot_scatter(
        master_runs,
        "final_entropy",
        "total_clusters",
        "Final entropy",
        "Total clusters",
        "Entropy-cluster relationship",
        scatter_dir / "02_entropy_vs_total_clusters.png",
    )
    plot_scatter(
        master_runs,
        "final_drift",
        "final_moves",
        "Final drift",
        "Final moves",
        "Drift-mobility relationship",
        scatter_dir / "03_drift_vs_final_moves.png",
    )

    manifest = out_dir / "FIGURE_GUIDE.txt"
    manifest.write_text(
        "\n".join([
            "Recommended main-thesis figures:",
            "01_core_scaling/01_final_entropy_vs_size.png",
            "01_core_scaling/02_final_gini_vs_size.png",
            "01_core_scaling/03_final_drift_vs_size.png",
            "01_core_scaling/05_final_move_rate_vs_size.png",
            "01_core_scaling/06_total_clusters_vs_size.png",
            "01_core_scaling/08_largest_cluster_fraction_by_species.png",
            "",
            "Representative dynamics figures:",
            "02_dynamics/Lxxx_01_moves_over_time.png",
            "02_dynamics/Lxxx_03_tau_over_time.png",
            "02_dynamics/Lxxx_04_entropy_over_time.png",
            "02_dynamics/Lxxx_06_drift_over_time.png",
            "",
            "Good appendix/robustness figures:",
            "04_seed_distributions/*.png",
            "05_heatmaps/*.png",
            "06_relationships/*.png",
            "",
            "Cluster morphology figures:",
            "03_clusters/*.png",
        ]),
        encoding="utf-8",
    )

    print(f"[done] figures written to: {out_dir}")
    print(f"[done] representative sizes used: {rep_sizes}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create all necessary thesis plots for the segregation experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("thesis_figures"))
    args = parser.parse_args()

    create_all_figures(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()