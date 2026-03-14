from __future__ import annotations

from pathlib import Path
import subprocess
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(".")
FRAME_DIR = BASE_DIR / "outputs" / "size_sweep" / "frames"
DATA_DIR = BASE_DIR/"outputs"/"size_sweep"/"analysis"
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape.exe")

REP_SIZE_MAIN = 100
REP_SEED_MAIN = 42

SNAPSHOT_CONDITIONS = [
    {
        "label": "Baseline\n(K=1)",
        "variant": "baseline",
        "layers": 1,
        "similarity": "cosine",
    },
    {
        "label": "Without influencer\n(K=3)",
        "variant": "without_influencer",
        "layers": 3,
        "similarity": "hybrid",
    },
    {
        "label": "With influencer\n(K=1)",
        "variant": "with_influencer",
        "layers": 1,
        "similarity": "hybrid",
    },
]

COND_ORDER = [
    "baseline",
    "without_influencer_L1",
    "without_influencer_L2",
    "without_influencer_L3",
    "with_influencer_L1",
    "with_influencer_L2",
    "with_influencer_L3",
]

COND_LABEL_MAP = {
    "baseline": "Baseline",
    "without_influencer_L1": "No influencer, K=1",
    "without_influencer_L2": "No influencer, K=2",
    "without_influencer_L3": "No influencer, K=3",
    "with_influencer_L1": "With influencer, K=1",
    "with_influencer_L2": "With influencer, K=2",
    "with_influencer_L3": "With influencer, K=3",
}

SMALL_SIZES_FOR_TABLE = [40, 100, 140]


# ============================================================
# Helpers
# ============================================================

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


def ordered_conditions_present(df: pd.DataFrame) -> list[str]:
    present = set(df["condition"].dropna().astype(str))
    base = [c for c in COND_ORDER if c in present]
    extra = sorted([c for c in present if c not in set(base)])
    return base + extra


def representative_sizes(all_sizes: list[int]) -> list[int]:
    all_sizes = sorted(set(int(x) for x in all_sizes))
    preferred = [40, 100, 140]
    selected = [s for s in preferred if s in all_sizes]
    if len(selected) >= 3:
        return selected
    fallback = [all_sizes[0], all_sizes[len(all_sizes)//2], all_sizes[-1]]
    for s in fallback:
        if s not in selected:
            selected.append(s)
    return selected[:3]


def fmt_mean_std(mean_val, std_val, ndigits=3) -> str:
    if pd.isna(mean_val):
        return "--"
    if pd.isna(std_val):
        return f"{mean_val:.{ndigits}f}"
    return f"{mean_val:.{ndigits}f} $\\pm$ {std_val:.{ndigits}f}"


def variant_label(v: str) -> str:
    return {
        "baseline": "Baseline",
        "without_influencer": "Without influencer",
        "with_influencer": "With influencer",
    }.get(v, v)


def add_shared_legend(fig, ax_for_handles, ncol=3, y=0.995):
    handles, labels = ax_for_handles.get_legend_handles_labels()
    labels = [COND_LABEL_MAP.get(l, l) for l in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=True,
        bbox_to_anchor=(0.5, y),
    )


# ============================================================
# Snapshot figure (from raw SVG frames using Inkscape)
# ============================================================

def find_available_steps(variant: str, size: int, seed: int, layers: int, similarity: str) -> list[int]:
    pattern = re.compile(
        rf"^{re.escape(variant)}_size{size}_seed{seed}_L{layers}_{re.escape(similarity)}_step(\d+)\.svg$"
    )
    steps = []
    for p in FRAME_DIR.glob("*.svg"):
        m = pattern.match(p.name)
        if m:
            steps.append(int(m.group(1)))
    steps = sorted(set(steps))
    if not steps:
        raise FileNotFoundError(
            f"No SVG files found for {variant}, size={size}, seed={seed}, layers={layers}, similarity={similarity}"
        )
    return steps


def choose_three_steps(steps: list[int]) -> list[int]:
    if len(steps) < 3:
        raise ValueError(f"Need at least 3 steps, found: {steps}")
    return [steps[0], steps[len(steps)//2], steps[-1]]


def svg_filename(variant: str, size: int, seed: int, layers: int, similarity: str, step: int) -> str:
    return f"{variant}_size{size}_seed{seed}_L{layers}_{similarity}_step{step}.svg"


def png_filename_from_svg(svg_name: str) -> str:
    return svg_name.replace(".svg", ".png")


def convert_svg_to_png(svg_path: Path, png_path: Path):
    if not INKSCAPE.exists():
        raise FileNotFoundError(f"Inkscape not found at: {INKSCAPE}")
    cmd = [
        str(INKSCAPE),
        str(svg_path),
        "--export-type=png",
        f"--export-filename={png_path}",
    ]
    subprocess.run(cmd, check=True)


def build_snapshot_rows():
    rows = []
    for cond in SNAPSHOT_CONDITIONS:
        steps = find_available_steps(
            cond["variant"], REP_SIZE_MAIN, REP_SEED_MAIN, cond["layers"], cond["similarity"]
        )
        chosen = choose_three_steps(steps)
        png_paths = []
        for step in chosen:
            svg_name = svg_filename(
                cond["variant"], REP_SIZE_MAIN, REP_SEED_MAIN, cond["layers"], cond["similarity"], step
            )
            svg_path = FRAME_DIR / svg_name
            png_path = FIG_DIR / png_filename_from_svg(svg_name)
            if not png_path.exists():
                convert_svg_to_png(svg_path, png_path)
            png_paths.append(png_path)

        rows.append({"label": cond["label"], "steps": chosen, "files": png_paths})
    return rows


def make_snapshot_figure():
    rows = build_snapshot_rows()
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)

    col_labels = ["Initial", "Midpoint", "Final"]

    for i, row in enumerate(rows):
        for j, png_path in enumerate(row["files"]):
            img = Image.open(png_path).convert("RGBA")
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

            if i == 0:
                axes[i, j].set_title(f"{col_labels[j]}\n(step {row['steps'][j]})", fontsize=12, pad=10)

            if j == 0:
                axes[i, j].text(
                    -0.08,
                    0.5,
                    row["label"],
                    transform=axes[i, j].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=12,
                )

    fig.suptitle(
        f"Representative spatial evolution at L={REP_SIZE_MAIN}, seed={REP_SEED_MAIN}",
        fontsize=15,
        y=1.02,
    )

    out_png = FIG_DIR / "fig_snapshots_L100_seed42.png"
    out_pdf = FIG_DIR / "fig_snapshots_L100_seed42.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Summary plots
# ============================================================

def add_derived_columns(master_runs, final_summary, step_summary):
    if {"total_red", "total_green", "total_blue"}.issubset(master_runs.columns):
        master_runs["occupied_final"] = master_runs["total_red"] + master_runs["total_green"] + master_runs["total_blue"]
        master_runs["final_move_rate"] = master_runs["final_moves"] / master_runs["occupied_final"].replace(0, np.nan)
        master_runs["mean_cluster_size"] = master_runs["occupied_final"] / master_runs["total_clusters"].replace(0, np.nan)

    if {"total_red_mean", "total_green_mean", "total_blue_mean"}.issubset(final_summary.columns):
        final_summary["occupied_final_mean"] = (
            final_summary["total_red_mean"] + final_summary["total_green_mean"] + final_summary["total_blue_mean"]
        )
        final_summary["final_move_rate_mean"] = (
            final_summary["final_moves_mean"] / final_summary["occupied_final_mean"].replace(0, np.nan)
        )
        final_summary["mean_cluster_size_mean"] = (
            final_summary["occupied_final_mean"] / final_summary["total_clusters_mean"].replace(0, np.nan)
        )

    if {"size", "moves_mean"}.issubset(step_summary.columns):
        step_summary["occupied_expected"] = 0.78 * (step_summary["size"] ** 2)
        step_summary["moves_rate_mean"] = step_summary["moves_mean"] / step_summary["occupied_expected"].replace(0, np.nan)


def plot_line_with_ci(ax, df, xcol, ycol, yci, ylabel, title=None, logy=False):
    for cond in ordered_conditions_present(df):
        g = df[df["condition"] == cond].sort_values(xcol)
        if g.empty or ycol not in g.columns:
            continue
        ax.plot(g[xcol], g[ycol], marker="o", linewidth=2, markersize=5, label=cond)
        if yci and yci in g.columns:
            lower = g[ycol] - g[yci]
            upper = g[ycol] + g[yci]
            ax.fill_between(g[xcol], lower, upper, alpha=0.15)

    ax.set_xlabel("Grid size $L$" if xcol == "size" else "Simulation step")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=14, pad=8)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)


def make_scaling_figure(final_summary):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_line_with_ci(
        axes[0, 0], final_summary, "size", "final_entropy_mean", "final_entropy_ci95",
        "Final entropy", "(a) Final entropy"
    )
    plot_line_with_ci(
        axes[0, 1], final_summary, "size", "final_gini_mean", "final_gini_ci95",
        "Final Gini impurity", "(b) Final Gini impurity"
    )
    plot_line_with_ci(
        axes[1, 0], final_summary, "size", "final_drift_mean", "final_drift_ci95",
        "Final drift", "(c) Final drift"
    )
    plot_line_with_ci(
        axes[1, 1], final_summary, "size", "final_move_rate_mean", None,
        "Final move rate", "(d) Final move rate", logy=True
    )

    add_shared_legend(fig, axes[0, 0], ncol=3, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = FIG_DIR / "fig_scaling.png"
    out_pdf = FIG_DIR / "fig_scaling.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_dynamics_figure(step_summary, size=100, out_name="fig_dynamics_L100"):
    df = step_summary[step_summary["size"] == size].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_line_with_ci(
        axes[0, 0], df, "step", "moves_mean", "moves_ci95",
        "Movement", "(a) Movement", logy=True
    )
    plot_line_with_ci(
        axes[0, 1], df, "step", "tau_mean", "tau_ci95",
        r"Adaptive threshold $\tau_t$", "(b) Adaptive threshold"
    )
    plot_line_with_ci(
        axes[1, 0], df, "step", "entropy_mean", "entropy_ci95",
        "Local entropy", "(c) Local entropy"
    )
    plot_line_with_ci(
        axes[1, 1], df, "step", "drift_mean", "drift_ci95",
        "Embedding drift", "(d) Embedding drift"
    )

    add_shared_legend(fig, axes[0, 0], ncol=3, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = FIG_DIR / f"{out_name}.png"
    out_pdf = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cluster_figure(final_summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    plot_line_with_ci(
        axes[0], final_summary, "size", "total_clusters_mean", "total_clusters_ci95",
        "Total clusters", "(a) Total clusters", logy=True
    )
    plot_line_with_ci(
        axes[1], final_summary, "size", "mean_cluster_size_mean", None,
        "Mean cluster size", "(b) Mean cluster size", logy=True
    )

    add_shared_legend(fig, axes[0], ncol=3, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_png = FIG_DIR / "fig_clusters.png"
    out_pdf = FIG_DIR / "fig_clusters.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_robustness_figure(master_runs):
    rep_sizes = representative_sizes(sorted(master_runs["size"].dropna().astype(int).unique().tolist()))
    df = master_runs[master_runs["size"].isin(rep_sizes)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    metrics = [
        ("final_entropy", "Final entropy"),
        ("final_moves", "Final movement"),
        ("final_drift", "Final drift"),
        ("total_clusters", "Total clusters"),
    ]

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        groups = []
        labels = []
        for s in rep_sizes:
            dsize = df[df["size"] == s]
            for cond in ordered_conditions_present(dsize):
                vals = dsize[dsize["condition"] == cond][metric].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                groups.append(vals)
                short = COND_LABEL_MAP.get(cond, cond)
                short = short.replace("No influencer", "No inf.")
                short = short.replace("With influencer", "With inf.")
                labels.append(f"{short}\nL={s}")

        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_title(title, fontsize=14, pad=8)
        ax.tick_params(axis="x", rotation=70, labelsize=9)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    #test

    out_png = FIG_DIR / "fig_robustness.png"
    out_pdf = FIG_DIR / "fig_robustness.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cluster_ccdf_figure(cluster_long, size=140):
    df = cluster_long[(cluster_long["size"] == size)].dropna(subset=["cluster_size"]).copy()
    if df.empty:
        return

    species_list = ["red", "green", "blue"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, species in zip(axes, species_list):
        dspec = df[df["species"] == species]
        for cond in ordered_conditions_present(dspec):
            vals = dspec[dspec["condition"] == cond]["cluster_size"].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            vals = np.sort(vals)
            ccdf = 1.0 - np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ccdf, marker=".", linestyle="-", linewidth=1.6, label=cond)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(species.capitalize(), fontsize=14, pad=8)
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CCDF")
        ax.grid(alpha=0.2)

    add_shared_legend(fig, axes[0], ncol=3, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out_png = FIG_DIR / "fig_cluster_ccdf_L140.png"
    out_pdf = FIG_DIR / "fig_cluster_ccdf_L140.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LaTeX table generation
# ============================================================

def write_main_results_table(final_summary: pd.DataFrame):
    df = final_summary[final_summary["size"].isin(SMALL_SIZES_FOR_TABLE)].copy()
    df = df.sort_values(["size", "variant", "layers"])

    lines = []
    lines.append(r"\begin{tabularx}{\fulllength}{l c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Variant & $L$ & $K$ & Entropy & Gini & Drift & Final moves & Clusters \\")
    lines.append(r"\midrule")

    for size in SMALL_SIZES_FOR_TABLE:
        dsize = df[df["size"] == size]
        for _, row in dsize.iterrows():
            v = variant_label(row["variant"])
            k = int(row["layers"])
            entropy = fmt_mean_std(row["final_entropy_mean"], row.get("final_entropy_std", np.nan))
            gini = fmt_mean_std(row["final_gini_mean"], row.get("final_gini_std", np.nan))
            drift = fmt_mean_std(row["final_drift_mean"], row.get("final_drift_std", np.nan))
            moves = fmt_mean_std(row["final_moves_mean"], row.get("final_moves_std", np.nan), ndigits=2)
            clusters = fmt_mean_std(row["total_clusters_mean"], row.get("total_clusters_std", np.nan), ndigits=1)

            lines.append(
                rf"{v} & {int(size)} & {k} & {entropy} & {gini} & {drift} & {moves} & {clusters} \\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines = lines[:-1]

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    out = TABLE_DIR / "table_main_results_compact.tex"
    out.write_text("\n".join(lines), encoding="utf-8")


def write_full_summary_table(final_summary: pd.DataFrame):
    df = final_summary.copy().sort_values(["size", "variant", "layers"])

    lines = []
    lines.append(r"\begin{tabularx}{\fulllength}{l c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Variant & $L$ & $K$ & Entropy & Gini & Drift & Final moves & Clusters \\")
    lines.append(r"\midrule")

    current_size = None
    for _, row in df.iterrows():
        size = int(row["size"])
        if current_size is not None and size != current_size:
            lines.append(r"\midrule")
        current_size = size

        v = variant_label(row["variant"])
        k = int(row["layers"])
        entropy = fmt_mean_std(row["final_entropy_mean"], row.get("final_entropy_std", np.nan))
        gini = fmt_mean_std(row["final_gini_mean"], row.get("final_gini_std", np.nan))
        drift = fmt_mean_std(row["final_drift_mean"], row.get("final_drift_std", np.nan))
        moves = fmt_mean_std(row["final_moves_mean"], row.get("final_moves_std", np.nan), ndigits=2)
        clusters = fmt_mean_std(row["total_clusters_mean"], row.get("total_clusters_std", np.nan), ndigits=1)

        lines.append(
            rf"{v} & {size} & {k} & {entropy} & {gini} & {drift} & {moves} & {clusters} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    out = TABLE_DIR / "table_full_final_summary.tex"
    out.write_text("\n".join(lines), encoding="utf-8")


def write_manifest():
    text = """
Main text assets
----------------
figures/fig_snapshots_L100_seed42.pdf
figures/fig_scaling.pdf
figures/fig_dynamics_L100.pdf
figures/fig_clusters.pdf
tables/table_main_results_compact.tex

Supplementary / appendix assets
-------------------------------
figures/fig_robustness.pdf
figures/fig_dynamics_L40.pdf
figures/fig_dynamics_L140.pdf
figures/fig_cluster_ccdf_L140.pdf
tables/table_full_final_summary.tex
""".strip()
    (BASE_DIR / "ASSET_MANIFEST.txt").write_text(text, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    master_runs = read_csv_required(DATA_DIR / "master_runs.csv")
    master_steps = read_csv_required(DATA_DIR / "master_steps.csv")
    final_summary = read_csv_required(DATA_DIR / "final_summary_by_variant_size_layers.csv")
    step_summary = read_csv_required(DATA_DIR / "step_summary_by_variant_size_layers.csv")
    cluster_long = read_csv_required(DATA_DIR / "final_cluster_distributions_long.csv")

    master_runs = add_condition_label(master_runs)
    master_steps = add_condition_label(master_steps)
    final_summary = add_condition_label(final_summary)
    step_summary = add_condition_label(step_summary)
    cluster_long = add_condition_label(cluster_long)

    add_derived_columns(master_runs, final_summary, step_summary)

    # figures
    make_snapshot_figure()
    make_scaling_figure(final_summary)
    make_dynamics_figure(step_summary, size=100, out_name="fig_dynamics_L100")
    make_dynamics_figure(step_summary, size=40, out_name="fig_dynamics_L40")
    make_dynamics_figure(step_summary, size=140, out_name="fig_dynamics_L140")
    make_cluster_figure(final_summary)
    make_robustness_figure(master_runs)
    make_cluster_ccdf_figure(cluster_long, size=140)

    # tables
    write_main_results_table(final_summary)
    write_full_summary_table(final_summary)

    # manifest
    write_manifest()

    print("Done. Figures written to:", FIG_DIR.resolve())
    print("Tables written to:", TABLE_DIR.resolve())
    print("Manifest written to:", (BASE_DIR / "ASSET_MANIFEST.txt").resolve())


if __name__ == "__main__":
    main()