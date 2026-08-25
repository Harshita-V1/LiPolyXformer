# LiPolyXformer

Official implementation of **LiPolyXformer: A Lightweight Spatial-Spectral Transformer with Polynomial Unmixing for Hyperspectral Anomaly Detection**.

The repository contains the implementation used for hyperspectral anomaly detection experiments on the **MUUFL Gulfport (MUUFL)** and **Pavia** datasets.

## Overview

LiPolyXformer is a lightweight reconstruction-based hyperspectral anomaly detection framework combining:

- Dual-branch spatial and spectral self-attention
- Bidirectional cross-attention with gated fusion
- An abundance-like bottleneck with 10 pseudo-endmembers
- A learnable polynomial spectral mixer
- Depthwise 1-D coefficient smoothing
- Per-band uncertainty estimation
- Uncertainty-aware reconstruction and patch aggregation
- NLL, spectral-angle, total-variation, and abundance sparsity losses

The paper describes the model as a shallow spatial-spectral transformer designed to reduce memory, runtime, and energy requirements while retaining competitive anomaly-detection performance.

## Repository structure

```text
LiPolyXformer/
├── LiPolyXformer_muufl.py
├── LiPolyXformer_pavia.py
├── AUC_muufl.py
├── AUC_pavia.py
├── resource_runner_muufl.py
├── resource_runner_pavia.py
├── main.ipynb
├── MUUFL_split.mat
├── pavia_split.mat
├── requirements.txt
└── README.md
```
Note: Dataset-specific Python implementations are provided for MUUFL and Pavia, with minor parameter and configuration adjustments as required by each dataset. The corresponding files follow the same overall architecture and implementation.

## Requirements

Python 3.9+ is recommended.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

For GPU experiments, install a PyTorch build compatible with your CUDA installation. The exact PyTorch/CUDA combination depends on the local system.

## Datasets

The repository uses preprocessed split files:

```text
MUUFL_split.mat
pavia_split.mat
```

The paper uses a 100 × 100 spatial crop for both datasets and overlapping 10 × 10 patches with stride 5.

The supplied split files are expected by the training scripts and should be kept in the same directory as the corresponding Python scripts unless an alternative path is passed through `--mat`.

## Training

### Pavia

Run the proposed LiPolyXformer configuration:

```bash
python LiPolyXformer_pavia.py \
    --mat pavia_split.mat \
    --epochs 20 \
    --lr 0.005 \
    --out_dir results_pavia/Proposed
```

### MUUFL

```bash
python LiPolyXformer_muufl.py \
    --mat MUUFL_split.mat \
    --epochs 20 \
    --lr 0.005 \
    --out_dir results_muufl/Proposed
```

The default number of epochs in both scripts is 20.

## Ablation experiments

The training scripts provide command-line switches for the ablation configurations described in the paper.

### No polynomial mixer

```bash
python LiPolyXformer_pavia.py --mat pavia_split.mat --epochs 20 --no_poly --out_dir results_pavia/NoPoly
```

### No cross-attention
--no_cross 

### No spatial attention
--attn_spa_off 

### No spectral attention
 --attn_spec_off

### No uncertainty head
 --no_uncertainty 

The same switches are available in `LiPolyXformer_muufl.py`.

## Resource monitoring

The repository contains separate monitoring utilities:

```text
resource_runner_muufl.py
resource_runner_pavia.py
```

These utilities record resource-related measurements such as runtime, CPU/GPU information, memory, power, and energy-related quantities used in the experimental analysis.

The notebook currently contains two bash cells that construct commands using these runners. In the embedded Python snippets, the import should refer to the dataset-specific runner:

```python
from resource_runner_pavia import run_with_monitor
```

for Pavia, and:

```python
from resource_runner_muufl import run_with_monitor
```

for MUUFL.

## Notebook

`main.ipynb` contains the experiment commands for the ablation configurations and the proposed model.

## Evaluation

The repository includes:

```text
AUC_muufl.py
AUC_pavia.py
```

for generating the ROC/AUC-related evaluation outputs from the saved experiment results. Please use these only as the correct version of measuring.

## Experimental configuration

The paper reports the following main configuration:

| Component | Setting |
|---|---|
| Pseudo-endmembers | 10 |
| Spatial MHSA heads | 6 |
| Spectral MHSA heads | 4 |
| Cross-attention heads | 4 |
| Patch tokens | 100 |
| Spatial MLP hidden dimension | 10 |
| Spectral MLP hidden dimension | 100 |
| Polynomial mixer | p + a2 p² + a3 p³ |
| Coefficient smoothing | Depthwise 1-D convolution |
| Uncertainty head | Linear-GELU-Linear |
| Loss | NLL + SAM + TV + L1 |
| Epochs | 20 |
| Learning rate | 5 × 10⁻³ |

The MUUFL training script currently uses dataset-specific command-line defaults for some architectural settings; check the script arguments when reproducing a particular experiment.

## Reproducibility

Random seeds can be controlled through:

```bash
--seed 42
```

The scripts save model outputs and experiment results under the directory supplied through:

```bash
--out_dir
```

For example:

```text
results_pavia/
└── Proposed/
    └── ...
```

## Citation

If you use this implementation in academic work, please cite the associated LiPolyXformer paper.

## Note on preprocessing

The paper states that the preprocessing pipeline described in reference [20] is used for patch extraction and dataset formatting for comparison and reproducibility. The proposed LiPolyXformer architecture and its spatial-spectral fusion, polynomial mixing, and uncertainty components are described as the contributions of this work.

Reference [20]:

Z. Wu and B. Wang, "Transformer-based autoencoder framework for nonlinear hyperspectral anomaly detection," *IEEE Transactions on Geoscience and Remote Sensing*, 62, 2024.
