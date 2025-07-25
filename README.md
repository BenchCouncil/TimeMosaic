# TimeMosaic

**TimeMosaic: Information-Density Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding**

![Framework](./figure/framework.png)

> This repository contains the official implementation of **TimeMosaic**, a novel framework for multivariate time series forecasting. It dynamically partitions input sequences into variable-length patches based on local temporal information density, and performs segment-wise forecasting through prompt-guided multi-task learning.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from Time-Series-Library. [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:
```bash
bash scripts/TimeMosaic/ETTh1.sh
```


## Features

- **Adaptive Patch Embedding**: Segment the input based on local information density with variable granularity.
- **Segment-wise Prompt Tuning**: Model asymmetric forecasting difficulty across time using learnable prompts.
- **Temporal Coherence**: Ensure each timestep belongs to exactly one patch, avoiding overlap and misalignment.
- **Unified Benchmark**: Evaluate across **17 real-world datasets** and **20+ baseline models** with consistent settings.
- **Visualization Tools**: Includes scripts for patch distribution and attention heatmap analysis.

## Included Models

Our benchmark integrates over **20+ state-of-the-art time series forecasting models** spanning various design paradigms, including Transformer-based, MLP-based, and hybrid architectures. All models are implemented under a unified codebase with consistent settings to ensure fair and reproducible comparisons.

The following models have been included in our evaluation suite:

- ✅ **TimeMosaic** (Ours): A novel framework combining adaptive patch granularity and segment-wise prompt tuning.
- ✅ **SimpleTM** (ICLR 2025): A strong baseline using wavelet transform and simplified self-attention.
- ✅ **TimeFilter** (ICML 2025): Performs fine-grained attention filtering via patch-wise gating.
- ✅ **xPatch** : xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition [[AAAI 2025]](https://arxiv.org/pdf/2412.17323).
- ✅ **DUET** (KDD 2025): Dual clustering enhanced transformer for modeling global-local structures.
- ✅ **PathFormer** (ICLR 2024): Utilizes adaptive patch selection guided by importance scores.
- ✅ **PatchMLP** (AAAI 2025): MLP-based patch modeling with multi-scale representations.
- ✅ **iTransformer** (ICLR 2024): Inverted attention across variables with strong long-term capacity.
- ✅ **PatchTST** (ICLR 2023): A pioneering patch-based time series transformer using channel-wise attention.
- ✅ **TimesNet** (ICLR 2023): Applies 2D variation modeling for general time series tasks.
- ✅ **DLinear** (AAAI 2023): A lightweight linear model shown to outperform many deep models.
- ✅ **TimeMixer / TimeMixer++** (ICLR 2024/2025): Decomposable multiscale mixing with multi-resolution attention.
- ✅ **MICN** (ICLR 2023): Joint modeling of multi-scale global and local context.
- ✅ **FreTS** (NeurIPS 2023): Frequency-domain MLPs for efficient sequence modeling.
- ✅ **Crossformer** (ICLR 2023): Captures cross-dimension dependencies across channels.
- ✅ **TiDE** (arXiv 2023): Dense encoder for long-term forecasting with full receptive field.
- ✅ **LightTS** (arXiv 2022): Efficient MLP structure tailored for multivariate prediction.
- ✅ **Autoformer** (NeurIPS 2021): Early decomposition-based forecasting Transformer.
- ✅ **FEDformer** (ICML 2022): Frequency-enhanced variant of the decomposition family.
- ✅ **Informer** (AAAI 2021): Utilizes ProbSparse attention for efficient long sequence modeling.
- ✅ **Pyraformer** (ICLR 2022): Introduces pyramidal structure to model long-range dependencies.
- ✅ **Reformer** (ICLR 2020): Low-memory attention with locality-sensitive hashing.
- ✅ **ETSformer** (arXiv 2022): Combines exponential smoothing with Transformer backbones.

All models are trained under identical lookback lengths and evaluation metrics (MSE/MAE), and the same `drop_last=False` setting to ensure fairness and reproducibility.


## Acknowledgements