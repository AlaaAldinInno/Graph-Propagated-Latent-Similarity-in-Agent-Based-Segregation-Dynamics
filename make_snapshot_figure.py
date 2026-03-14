from pathlib import Path
import subprocess
import re

import matplotlib.pyplot as plt
from PIL import Image


INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape.exe")
FRAME_DIR = Path("outputs/size_sweep/frames")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 100
SEED = 42

CONDITIONS = [
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

COL_LABELS = ["Initial", "Midpoint", "Final"]


def find_available_steps(variant: str, size: int, seed: int, layers: int, similarity: str):
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


def choose_three_steps(steps):
    if len(steps) < 3:
        raise ValueError(f"Need at least 3 steps, found only: {steps}")
    first = steps[0]
    middle = steps[len(steps) // 2]
    last = steps[-1]
    return [first, middle, last]


def svg_name(variant: str, size: int, seed: int, layers: int, similarity: str, step: int):
    return f"{variant}_size{size}_seed{seed}_L{layers}_{similarity}_step{step}.svg"


def png_name(svg_filename: str):
    return svg_filename.replace(".svg", ".png")


def convert_svg_to_png(svg_path: Path, png_path: Path):
    cmd = [
        str(INKSCAPE),
        str(svg_path),
        "--export-type=png",
        f"--export-filename={png_path}",
    ]
    subprocess.run(cmd, check=True)


def prepare_images():
    rows = []

    if not INKSCAPE.exists():
        raise FileNotFoundError(f"Inkscape not found at: {INKSCAPE}")

    for cond in CONDITIONS:
        steps = find_available_steps(
            cond["variant"], SIZE, SEED, cond["layers"], cond["similarity"]
        )
        chosen_steps = choose_three_steps(steps)

        row_files = []
        for step in chosen_steps:
            svg_file = svg_name(
                cond["variant"], SIZE, SEED, cond["layers"], cond["similarity"], step
            )
            svg_path = FRAME_DIR / svg_file
            png_path = FIG_DIR / png_name(svg_file)

            if not png_path.exists():
                convert_svg_to_png(svg_path, png_path)
                print(f"Created {png_path}")
            else:
                print(f"Already exists: {png_path}")

            row_files.append(png_path)

        rows.append(
            {
                "label": cond["label"],
                "steps": chosen_steps,
                "files": row_files,
            }
        )

    return rows


def load_image(path: Path):
    return Image.open(path).convert("RGBA")


def make_panel(rows):
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)

    for i, row in enumerate(rows):
        for j, img_path in enumerate(row["files"]):
            img = load_image(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

            if i == 0:
                axes[i, j].set_title(f"{COL_LABELS[j]}\n(step {row['steps'][j]})", fontsize=12, pad=10)

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

    fig.suptitle(f"Representative spatial evolution at L={SIZE}, seed={SEED}", fontsize=15, y=1.02)

    out_png = FIG_DIR / "fig_snapshots_L100_seed42.png"
    out_pdf = FIG_DIR / "fig_snapshots_L100_seed42.pdf"

    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    rows = prepare_images()
    make_panel(rows)
