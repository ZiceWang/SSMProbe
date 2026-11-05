# SSMProbe

**Learning Dynamic Evidence Routes for Vision Transformer Probing**

**Accepted by [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/156006).**

Zice Wang and Zhenyu Zhang

[![NeurIPS 2026 Accepted](https://img.shields.io/badge/NeurIPS_2026-Accepted-68468C?style=flat-square)](https://neurips.cc/virtual/2026/poster/156006)
[![arXiv](https://img.shields.io/badge/arXiv-2605.00915v3-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00915v3)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ZiceWang/SSMProbe)
[![Project page](https://img.shields.io/badge/Project_Page-Website-2563EB?style=flat-square&logo=react&logoColor=white)](https://zicewang.github.io/SSMProbe/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=FFD43B)](#environment)

SSMProbe learns a Sinkhorn route through the patch tokens of a frozen vision transformer and reads that sequence with a diagonal S4 decoder. The fixed sequential coefficients make route organization and readout behavior inspectable.

## Repository layout

```text
SSMProbe/
  backbone/       Frozen MAE, BEiT, DINOv2, and supervised ViT backbones
  config.py       Dataset, backbone, routing, head, and training configuration
  dataset/        Dataset loading and feature extraction
  experiments/    Training, ablations, and diagnostic programs
  head/           Probe heads and deterministic scan orders
  plotting/       Visualization programs
  routing/        Differentiable token-routing operators
  trainer/        Probe training and evaluation
  utils/          FLOP analysis
s4/               Local torch-s4 implementation and tests
scripts/          Result summaries and plotting utilities
```

The [project homepage](https://zicewang.github.io/SSMProbe/) is built with React + Vite. Its source is maintained independently on the `feat/project-page` branch; GitHub Pages publishes the compiled site from `docs/pages`.

## Environment

Use Python 3.10 or later and a PyTorch build appropriate for your hardware. For example, with [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install -e ./s4
```

For CUDA, install the appropriate PyTorch build using the [official instructions](https://pytorch.org/get-started/locally/). Keep dataset access credentials in your local Hugging Face configuration or environment; never put them in source files.

## Running experiments

Run commands from the repository root. Each experiment exposes its own options; check its source and `--help` before launching a run.

```sh
uv run --no-project python -m SSMProbe.experiments.main_experiment --help
uv run --no-project python -m SSMProbe.experiments.main_experiment --seeds 42,43,44,45,46
```

The main experiment script uses the settings in `make_main_config`. Dataset access and model downloads are handled by Hugging Face. Match configuration, checkpoint selection, patch order, and training budget to the protocol of the result you intend to reproduce; the main script's defaults are not a substitute for the paper's experiment-specific protocols.

Test the bundled S4 implementation with:

```sh
uv pip install pytest
uv run --no-project python -m pytest s4/tests
```

Experiment outputs, checkpoints, raw logs, private manuscripts, and internal review records are excluded from version control. Analysis scripts accept local output/checkpoint paths; generate the corresponding artifacts before using them.

## Citation

```bibtex
@article{wang2026learning,
  title   = {Learning Dynamic Evidence Routes for Vision Transformer Probing},
  author  = {Wang, Zice and Zhang, Zhenyu},
  journal = {arXiv preprint arXiv:2605.00915},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.00915v3}
}
```
