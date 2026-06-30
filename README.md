# pdac_tumoroid

Code accompanying:

> **An integrated platform for high-throughput phenospace learning of 3D multilineage organoid systems**
> Okuda R, Harmel C, Xu Q, Mary H, Schulz P, Steinacher L, D'Arcangelo E, Gjeta B, Signer M, Cubela I, Bickle M, Lutolf MP, Cabon L, Lukonin I, Camp G.
> *bioRxiv* (2025). doi:[10.1101/2025.11.27.690952](https://doi.org/10.1101/2025.11.27.690952)

This repository contains the experimental–computational pipeline for **high-throughput spatial phenotyping of 3D multilineage organoids**. The system pairs a 384-well pancreatic ductal adenocarcinoma (PDAC) + cancer-associated fibroblast (CAF) tumoroid culture with multiplexed imaging, then learns a quantitative "phenospace" that preserves spatial architecture. We use it to identify pathway modulators of the fibrotic microenvironment and stroma-dependent vulnerabilities, and extend the platform to immune-competent systems and patient-derived explants.

## Overview

The pipeline runs in three stages:

1. **Image processing** (`image_processing/`) — turn raw multiplexed microscopy into single-object features.
2. **Phenospace learning** (`phenocoder/`, `laminator/`) — embed objects into a learned latent space and characterize spatial organization.
3. **Analysis & figures** (`analysis/`) — reproduce the screens, benchmarks, and figures from the paper.

Configuration is YAML-driven (`configs/`), and the heavy compute steps are dispatched to HPC schedulers via job scripts (`bsub/` for LSF, `sbatch/` for Slurm).

## Setup

The full software environment (Python + R/Bioconductor) is frozen in [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml   # creates the `pdac_tumoroid` environment
conda activate pdac_tumoroid
```

Run scripts from the repository root so the local packages (`analysis`, `phenocoder`, `image_processing`, `benchmarking`, `laminator`) are importable. GPU (CUDA) is required for Cellpose segmentation and CVAE training.

## Repository layout

| Path | Description |
|------|-------------|
| `image_processing/` | CLI-driven microscopy pipeline: flatfield & illumination correction, background correction, MIP, 3D stitching, cross-cycle registration, Cellpose segmentation, feature extraction, neighborhood construction, montaging, and QC. |
| `phenocoder/` | Core ML package (also see [standalone implementation of Phenocoder](https://github.com/devsystemslab/phenocoder)). Conditional/standard convolutional VAE, patch generators for nuclei or grid sampling, phenospace embedding, spatial graphs & convex-hull metrics via squidpy, clustering, training, and plotting. |
| `laminator/` | Reimplementation of Laminator ([Wahle et al., *Nat. Biotechnol.* 2023](https://www.nature.com/articles/s41587-023-01747-2)) for "laminating" organoid images into oriented windows and radial-neighborhood message passing. Used for length-scale analyses. |
| `benchmarking/` | Embedding-quality metrics: graph connectivity, multi-label AMI, cLISI/CAS/NASW, plus ARI/NMI, used to compare phenocoder embeddings against nuclei-only and alternative baselines. |
| `configs/` | Per-screen YAML configs and master parameters (screens, channel/marker maps, plates, phenocoder model params). |
| `bsub/`, `sbatch/` | Cluster job scripts for image processing, dataset generation, model training, phenocoding, and benchmarking. |
| `analysis/` | Numbered, per-experiment pipelines that reproduce the study (see below). |

## Analyses

Each subdirectory of `analysis/` is a self-contained, numerically ordered pipeline (`01_…`, `02_…`) mixing Python and R. Files prefixed `run_` perform computation; files prefixed `plot_` generate figures. Subdirectories are listed in the order their main figures appear in the paper.

| Subdirectory | Corresponds to |
|--------------|----------------|
| `brightfield_imaging/` | Brightfield imaging montages. |
| `z_undersample_and_segmentation_eval/` | Validation: z-stack undersampling and segmentation-quality assessment. |
| `length_scale_sensitivity/` | Laminator-based message passing and distance-scale sensitivity analysis. |
| `timecourse/` | Multilineage developmental timecourse — diffusion maps, MLP marker imputation, spatial prediction, latent-space traversal. |
| `pilotscreen/` | FDA-compound pilot screen, including cross-plate label transfer and spatial-graph rendering. |
| `multiple_patients/` | Generalization across multiple patient-derived lines. |
| `tumoroidscreen/` | Main PDAC + CAF inhibitor screen — phenocoding, benchmarking, feature heatmaps, compound-effect analysis, STRING network, example tumoroids. |

## Data availability

The processed images and `mdata` (`.h5mu`) files are deposited in the EMBL-EBI BioImage Archive:

> **DOI:** _TBD — add accession/DOI on publication._

Each screen is organized as `<screen>/images/` and `<screen>/mdata/`. The analysis scripts read these `.h5mu` files (see `analysis/`).

## License

MIT — see [LICENSE.md](LICENSE.md). © 2026 DevSystems Lab.
