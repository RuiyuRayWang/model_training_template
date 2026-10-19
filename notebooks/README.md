# Minuet

This project implements the Minuet model using PyTorch.

## Directory Structure

```
Minuet/
├── configs/            # Training configurations
├── checkpoints/        # Model checkpoints
├── data/               # Source data before preprocessing
├── datasets/           # Preprocessed datasets
├── models/             # Model definitions
├── notebooks/          # Jupyter notebooks for experiments
├── scripts/            # Training and evaluation scripts
├── utils/              # Utility functions
├── tmp/                # Deprecated files
└── README.md           # Project description
```

## Getting Started

1. Install dependencies:
   ```bash
   $ conda activate dp
   (dp) $ pip install -r requirements.txt 
   (dp) $ conda install bedtools
   ```

2. Preprocess data:
   Edit one config under `configs/` (for example `configs/config_dp_catlas_strpal_49k_12_reduced_11_topics_top3k_update.yaml`).

   The preprocess script supports two split modes via `dataset.split.mode`:
   - `random` (default): set `dataset.split.train_ratio`, `dataset.split.val_ratio`, `dataset.split.random_seed`
   - `chromosome`: set `dataset.split.chromosome.train|val|test` as chromosome lists
     - optional: `dataset.split.allow_missing_chromosomes: true` to ignore listed chromosomes that have zero regions (for example, `chrY` in some datasets)

   Run preprocessing from the project root:
   ```bash
   (dp)$ python -m scripts.preprocess -c configs/config_dp_catlas_strpal_49k_12_reduced_11_topics_top3k_update.yaml
   ```
   Equivalent direct script form:
   ```bash
   (dp)$ python scripts/preprocess.py -c configs/config_dp_catlas_strpal_49k_12_reduced_11_topics_top3k_update.yaml
   ```

3. Training with huggingface accelerate:
   ```bash
   screen
   cd DeepPallidum
   conda activate dp
   accelerate launch --num_processes 8 --main_process_port 29500 train_dp_catlas_strpal_top3k_accelerate.py
   ```
   Note: if multiple distributed jobs run on the same machine, set a different `--main_process_port` per job to avoid port conflicts.

   Ablation run scripts (A-E):
   - Run A (baseline): `train_dp_catlas_strpal_top3k_accelerate.py`
   - Run B (mean pooling): `train_dp_catlas_strpal_top3k_mean_pooling_accelerate.py`
   - Run C (attention pooling): `train_dp_catlas_strpal_top3k_attn_pooling_accelerate.py`
   - Run D (BCE + pos_weight): `train_dp_catlas_strpal_top3k_bce_posweight_accelerate.py`
   - Run E (CNN-only): `train_dp_catlas_strpal_top3k_cnn_only_accelerate.py`
   - Capacity-scaled Minuet attention model: `train_dp_catlas_strpal_top3k_minuet_capacity_accelerate.py`
     - Supports optional dilated-convolution stem via `train.model_kwargs.use_dilated_conv`, `conv_kernel_size`, and `dilations`.
   - Minuet dilation ablation control (vanilla conv): `train_dp_catlas_strpal_top3k_minuet_capacity_vanilla_accelerate.py`
   - Minuet dilation ablation treatment ([1,2,4]): `train_dp_catlas_strpal_top3k_minuet_capacity_dilated124_accelerate.py`
   - Minuet gated-dilation ablation treatment ([1,2,4] + token-wise branch gating): `train_dp_catlas_strpal_top3k_minuet_capacity_gated_dilated124_accelerate.py`
   - Minuet strengthened readout head (2-layer MLP + optional per-label calibration): `train_dp_catlas_strpal_top3k_minuet_readout_mlp2_accelerate.py`
   

4. Evaluate the model:
   Use the notebook [evaluation_template.ipynb](/home/ray/DeepPallidum/notebooks/evaluation_template.ipynb).

## Requirements

- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`
