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


## Acknowledgements