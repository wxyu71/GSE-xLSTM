# GSE-xLSTM

**GSE-xLSTM: Spatiotemporal Decoupled Multivariate Time Series Forecasting with sLSTM and Grouped Spatial MLP**

Authors: Wenxuan Yu, Yujuan Xu, Liming Jiang (corresponding author), and Shaomiao Chen
School of Computer Science and Engineering, Hunan University of Science and Technology, Xiangtan, China

This work has been accepted at PRICAI 2026.

## Citation

```bibtex
@inproceedings{yu2026gse-xlstm,
  title     = {GSE-xLSTM: Spatiotemporal Decoupled Multivariate Time Series Forecasting with sLSTM and Grouped Spatial MLP},
  author    = {Yu, Wenxuan and Xu, Yujuan and Jiang, Liming and Chen, Shaomiao},
  booktitle = {PRICAI 2026},
  year      = {2026}
}
```

## Overview

GSE-xLSTM is a grouped spatial-enhanced xLSTM based on a spatiotemporal decoupled dual-branch architecture for multivariate time series forecasting.

- **Temporal Branch**: Patch-sLSTM under the channel-independent paradigm extracts long-range temporal feature patterns at linear complexity.
- **Spatial Branch**: Projects variates into low-dimensional subspaces via a learnable grouping matrix for local mixing at linear time complexity.

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `xlstm==1.0.3`
- `torch`
- `lightning`

## Datasets

We evaluate on the following datasets. Please download them manually and place them under `dataset/` with the following structure:

```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
└── exchange_rate/
    └── exchange_rate.csv
```

Please download the datasets from their original sources and place them in the directory structure shown above.

## Reproducing Results

The `best_configs/` directory contains example configurations for ETTm1, ETTm2, and Electricity at prediction length 96. Each script runs training with 3 random seeds (2021, 2022, 2023).

```bash
# Example: ETTm1, prediction length 96
bash best_configs/ETTm1/pl96.sh 0  # GPU 0

# Example: Electricity, prediction length 96
bash best_configs/Electricity/pl96.sh 0
```

Full training configurations for all datasets and prediction horizons will be provided in future updates.

## Project Structure

```
GSE-xLSTM/
├── gse_xlstm/                    # Core source code
│   ├── models/
│   │   └── gse_xlstm.py          # GSE-xLSTM model (GSEXlstm)
│   ├── layers/                   # Custom layers
│   ├── exp/                      # Experiment management
│   ├── data_provider/            # Data loading
│   ├── lit/                      # Lightning integration
│   └── utils/                    # Utilities
└── best_configs/                 # Example configurations
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
