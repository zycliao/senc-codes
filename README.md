# [ECCV 2024] SENC: Handling Self-collision in Neural Cloth Simulation

[[Project]](https://zycliao.com/senc/) &nbsp; [[Arxiv]](https://arxiv.org/abs/2407.12479) &nbsp; [[Checkpoints]](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/zliao_connect_hku_hk/EaAG1zk7U1RMpvZ7AnzPcZkBOAW7KPSRE78MztTkbNHPPg?e=laLihg)

SENC is a neural cloth simulator that explicitly models **cloth self-collision**. Given a body-motion sequence (SMPL parameters) and a garment mesh, it rolls out a physically plausible garment animation that avoids garment–garment self-intersections. The network and graph representation build on top of [HOOD](https://github.com/dolorousrtur/hood); the key contribution of this repo is a differentiable **self-intersection energy** (`criterions/postcvpr/mataug/self_intersection_energy.py`) driven by a fast self-intersection detector (`utils_self_inter/`).

---

## Table of Contents
- [Repository structure](#repository-structure)
- [1. Environment setup](#1-environment-setup)
- [2. Data & checkpoints](#2-data--checkpoints)
- [3. Configure data paths](#3-configure-data-paths)
- [4. Inference](#4-inference)
- [5. Training](#5-training)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Repository structure

```
senc-codes/
├── train.py                 # Training entry point
├── inference.py             # Inference / rollout entry point
├── configs/
│   ├── senc.yaml            # Main SENC config (self-collision enabled)
│   └── hood.yaml            # HOOD baseline config
├── models/                  # Network + graph model definitions
├── runners/                 # Training / rollout loops
├── criterions/              # Physical energy terms (loss functions)
│   └── postcvpr/mataug/self_intersection_energy.py   # <-- SENC's core loss
├── datasets/                # Dataset / dataloader definitions
├── utils/                   # Meshes, SMPL, validation, IO, defaults, ...
│   └── defaults.py          # Data/project paths (from environment variables)
└── utils_self_inter/        # Self-intersection detection (needs a compiled module)
```

---

## 1. Environment setup

SENC requires a **CUDA-capable GPU** (the code hard-codes `cuda:0`). It was developed with Python 3.9 + PyTorch (CUDA). We recommend a fresh conda environment.

```bash
conda create -n senc python=3.9 -y
conda activate senc

# --- PyTorch (pick the CUDA build matching your machine) ---
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# --- PyTorch Geometric ---
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# --- PyTorch3D (used for nearest-neighbor / collision queries) ---
# See https://github.com/facebookresearch/pytorch3d for install options
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# --- SMPL body model ---
pip install smplx

# --- Remaining Python dependencies ---
pip install numpy pandas scipy scikit-learn networkx einops \
            omegaconf munch tqdm matplotlib tensorboardX wandb huepy
```

### libigl with CGAL

The self-intersection detector uses `igl.copyleft.cgal` (the CGAL-backed part of the
[libigl](https://libigl.github.io/libigl-python-bindings/) Python bindings):

```bash
pip install libigl
```

### Native self-intersection module (`cmake_example`)

`utils_self_inter/` imports a compiled C++/pybind11 module named **`cmake_example`**
(`import cmake_example`) that accelerates self-intersection queries. Its sources are in
`csrc/self_inter/`; build and install it into the active environment with a C++17
compiler (MSVC on Windows, gcc/clang on Linux):

```bash
pip install ./csrc/self_inter
```

It is needed whenever the config lists the `self_intersection_energy` criterion (as
`senc.yaml` does), including for inference, because all criteria in the config are
imported at start-up. Configs without that criterion (e.g. `hood.yaml`) run without it.

---

## 2. Data & checkpoints

SENC reuses the **HOOD data layout**. Download the HOOD data bundle by following the
instructions in the [HOOD repository](https://github.com/dolorousrtur/hood), and the
SENC checkpoints from the [checkpoint link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/zliao_connect_hku_hk/EaAG1zk7U1RMpvZ7AnzPcZkBOAW7KPSRE78MztTkbNHPPg?e=laLihg)
at the top of this page.

You need a **data root** directory (referred to below as `HOOD_DATA`) with roughly this layout:

```
HOOD_DATA/
├── aux_data/
│   ├── smpl/                     # SMPL_NEUTRAL.pkl, SMPL_FEMALE.pkl, ...
│   ├── garments_dict.pkl         # garment templates
│   └── datasplits/               # train.csv, val_sim.csv, ...
├── vto_dataset/
│   └── smpl_parameters/          # body-motion sequences (*.pkl, e.g. stretch.pkl)
├── trained_models/               # pretrained checkpoints (e.g. postcvpr.pth)
└── experiments/                  # created automatically for your training runs
```

Place the downloaded SENC checkpoint under `HOOD_DATA/trained_models/` (or
`HOOD_DATA/experiments/...`) and point the config / inference script at it (see below).

---

## 3. Configure data paths

Paths are read from environment variables in **`utils/defaults.py`**:

```bash
export HOOD_DATA=/path/to/hood_data        # the data root from step 2 (required)
export NC_DIR=/path/to/neural_cloth        # simulated ground truth for validation
                                           # (default: $HOOD_DATA/neural_cloth)
export HOOD_PROJECT=/path/to/senc-codes    # this repository (default: auto-detected)
```

`DEFAULTS` (data_root, experiment_root, vto_root, aux_data, ...) is derived from
`HOOD_DATA`. If `HOOD_DATA` is not set it defaults to `senc-codes/hood_data`.

---

## 4. Inference

`inference.py` loads a checkpoint, rolls out a garment over a body-motion sequence, and
saves the result as a point-cache (`.pc2`) plus a template `.obj` mesh that can be played
back in Blender / Houston / any DCC tool.

Because inference parameters are set **inline** at the top of `inference.py`, edit the
script rather than passing CLI flags. The key knobs:

```python
config_name    = 'senc'              # config in configs/ (must match the checkpoint)
garment_name   = 'dress'             # garment template name (from garments_dict.pkl)
checkpoint_path = HOOD_DATA/trained_models/step_0000070000.pth  # released SENC checkpoint
sequence_path  = .../smpl_parameters/stretch.pkl   # driving motion
# material parameters: density, lame_mu, lame_lambda, bending_coeff, wind, ...
```

Put your downloaded checkpoint at `HOOD_DATA/trained_models/step_0000070000.pth` (or change
`checkpoint_path`), then run:

```bash
python inference.py
```

Outputs are written to `HOOD_DATA/inference/<config_name>/` (`<mat>_<garment>.pc2` and
the corresponding mesh), and the raw trajectory
dictionary is pickled to `HOOD_DATA/temp/<save_name>.pkl`. The rollout length is
controlled by `n_steps` in the `runner.valid_rollout(...)` call.

---

## 5. Training

Training is launched through `train.py` and configured with a YAML file from `configs/`.
The config name is passed on the command line and selects the model, runner, dataset, and
active loss terms:

```bash
# Train the full SENC model (self-collision energy enabled)
python train.py --config senc

# Train the HOOD baseline (no self-collision energy)
python train.py --config hood
```

Useful notes:

- **Resume a run:** `python train.py --resume <experiment_dir_name>` — restores the latest
  checkpoint plus optimizer/scheduler state from `HOOD_DATA/experiments/<experiment_dir_name>/`.
- **Validation-only pass:** add `only_validation=true` on the command line.
- **Checkpoints & logs** are written to `HOOD_DATA/experiments/<config>_<timestamp>/`
  (a snapshot of the code is backed up there too). Metrics are logged via TensorBoard
  (`tensorboardX`) and optionally Weights & Biases (`wandb`).
- The self-collision loss weight and other physics coefficients live under the
  `criterions:` section of the config (e.g. `postcvpr.mataug.self_intersection_energy.weight`).
- Only **`batch_size: 1`** is supported (the model is applied per-node on the graph — see
  `datasets/README.md`).
- **Initialise from a checkpoint:** set `experiment.checkpoint_path` in the config (relative
  to `HOOD_DATA`); `senc.yaml` defaults to `null`, i.e. training from scratch.

> The `senc.yaml` config trains the `postcvpr_space_ext` model/runner with the
> `self_intersection_energy` criterion enabled (`self_col: True`). Removing that criterion
> from the config reproduces the collision-free HOOD-style baseline and lets you run
> without the `cmake_example` native module.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liao2024senc,
  title={SENC: Handling self-collision in neural cloth simulation},
  author={Liao, Zhouyingcheng and Wang, Sinan and Komura, Taku},
  booktitle={European Conference on Computer Vision},
  pages={385--402},
  year={2024},
  organization={Springer}
}
```

## Acknowledgements

This codebase is built on top of [HOOD](https://github.com/dolorousrtur/hood). We thank the
authors for releasing their code, dataset layout, and pretrained models. Please also refer
to and cite HOOD and the SMPL body model as appropriate.

## License

This code is released under the [MIT License](LICENSE). It includes code from
[HOOD](https://github.com/dolorousrtur/hood) (MIT License, Copyright (c) 2023 Artur Grigorev).
Note that the self-intersection detector depends on `igl.copyleft.cgal`, which is backed by
GPL-licensed [CGAL](https://www.cgal.org/) code, and that the SMPL body model and the HOOD
data are distributed under their own licenses.
