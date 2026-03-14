# Graph-Propagated Latent Similarity in Agent-Based Segregation Dynamics

Code and analysis pipeline for the paper on graph-propagated latent similarity in agent-based segregation dynamics. The repository contains a C++ simulation engine and Python utilities for experiment orchestration, aggregation, visualization, and paper-ready asset generation.

## Overview

This project studies segregation dynamics in an agent-based setting where relocation decisions are driven not only by local neighborhood composition, but also by graph-propagated latent representations. The implementation compares multiple experimental conditions:

- **Baseline**: a classical local-neighborhood condition using cosine-based similarity.
- **Without influencer**: graph-propagated latent similarity without influencer links.
- **With influencer**: graph-propagated latent similarity augmented with influencer connections.

Across these conditions, the code evaluates how model variant and propagation depth affect emergent spatial structure, convergence behavior, and cluster morphology.

## Repository structure

```text
.
├── cpp/
│   └── sgnn_runner.cpp              # Core C++ simulator and batch runner
├── scripts/
│   ├── run_experiments.py           # Builds the simulator and launches sweeps
│   └── plot_results.py              # Convenience wrapper for summarization/plotting
├── summarize_size_sweep.py          # Aggregates raw runs into analysis tables/plots
├── make_snapshot_figure.py          # Builds representative snapshot figures from SVG frames
├── make_thesis_figures.py           # Generates thesis/presentation-style figures
└── build_paper_assets.py            # Produces final paper figures, tables, and asset manifest
```

## Main outputs

The pipeline is organized around a size-sweep experiment and produces outputs under `outputs/size_sweep/`.

Expected generated artifacts include:

- `outputs/size_sweep/metrics/` — per-run stepwise metrics CSV files
- `outputs/size_sweep/frames/` — representative SVG snapshots of lattice states
- `outputs/size_sweep/analysis/` — merged and summarized CSV tables
- `outputs/size_sweep/plots/` — exploratory and summary visualizations
- `figures/` — paper-ready PDF figures
- `tables/` — LaTeX tables for manuscript insertion
- `ASSET_MANIFEST.txt` — list of final assets intended for the manuscript

## Metrics reported

The simulator tracks several quantities through time and at convergence:

- **moves**: number of agent relocations at each step
- **tau**: satisfaction threshold used during the step
- **entropy**: diversity/uncertainty in local latent composition
- **gini**: concentration/imbalance measure over latent mixture structure
- **drift**: representation change between successive steps
- **cluster statistics**: total clusters, per-species cluster counts, largest cluster size, and largest-cluster fractions

These metrics are aggregated by experimental condition, lattice size, propagation depth, and seed.

## Requirements

### Core requirements

- A C++17-compatible compiler
- CMake
- Python 3.10+

### Python packages

The Python scripts import the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`

Install them with:

```bash
pip install numpy pandas matplotlib pillow
```

### Optional dependency

`build_paper_assets.py` converts SVG snapshots to PNG using **Inkscape**. In the current script, the executable path is configured explicitly for Windows:

```python
INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape.exe")
```

Adjust this path for your system before building the final snapshot-based figures.

## Running the experiments

### 1. Build and execute the size sweep

From the repository root:

```bash
python scripts/run_experiments.py --workers 4 --min-size 20 --max-size 140 --step-size 20
```

This script:

1. configures the CMake build,
2. compiles the C++ simulator, and
3. launches the experiment sweep.

The simulator evaluates the main variants across multiple lattice sizes, seeds, and propagation depths.

### 2. Summarize raw outputs

```bash
python summarize_size_sweep.py
```

This step reads the raw metrics from `outputs/size_sweep/metrics/` and writes:

- merged step-level and run-level tables,
- aggregated summaries by variant, size, and layer,
- compact result tables for the paper, and
- analysis plots for scaling, dynamics, and cluster distributions.

### 3. Generate paper assets

```bash
python build_paper_assets.py
```

This script creates the final manuscript assets, including representative figures and LaTeX tables.

## Alternative figure-generation workflows

### Thesis-style figures

```bash
python make_thesis_figures.py --data-dir outputs/size_sweep/analysis --out-dir thesis_figures
```

Use this path when preparing broader presentation or thesis material rather than the final paper layout.

### Snapshot figure only

```bash
python make_snapshot_figure.py
```

Use this if you only want representative state snapshots from the generated SVG frames.

## Experimental design encoded in the simulator

The C++ runner enumerates the main comparison conditions:

- `baseline` with `L=1`
- `without_influencer` with `L=1,2,3`
- `with_influencer` with `L=1,2,3`

For non-baseline variants, the simulator uses graph propagation depth (`gcn_layers`) together with a hybrid similarity function. The baseline uses a cosine-based local criterion. The code also supports adaptive satisfaction thresholding through quantile-based `tau` selection.

## Reproducibility notes

- The run tag encodes `variant`, `size`, `seed`, `layers`, and `similarity`, which makes the output files traceable.
- Representative paper figures are built from summarized CSV outputs, so the recommended order is:
  1. run experiments,
  2. summarize outputs,
  3. build paper assets.
- Snapshot assets depend on the presence of SVG frames in `outputs/size_sweep/frames/`.
- Some scripts assume execution from the repository root.

## Typical workflow

```bash
python scripts/run_experiments.py --workers 4 --min-size 20 --max-size 140 --step-size 20
python summarize_size_sweep.py
python build_paper_assets.py
```

## Notes on this archive

This archive currently contains the simulation and analysis scripts. If you are packaging the public repository, make sure the final repo also includes the standard project infrastructure files you use locally, such as:

- `CMakeLists.txt`
- `requirements.txt` or `environment.yml`
- license information
- manuscript source and/or supplementary material, if intended for release

## Citation

If you use this code in academic work, please cite the associated paper once bibliographic details are finalized.

```bibtex
@article{graph_propagated_latent_similarity_segregation,
  title   = {Graph-Propagated Latent Similarity in Agent-Based Segregation Dynamics},
  author  = {Author(s) TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## Contact

For questions, issues, or collaboration related to the manuscript or codebase, please contact the repository author(s).
