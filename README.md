[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2410.21560)

# ImmunoHisto Benchmark

We assess performance of histopathology foundation models trained on H&E cancer 
dataset on OOD Immunohistochemistry (IHC) autoimmune datasets. 

## Models benchmarked

![models_benchmarked.png](figures/models_benchmarked.png)

## Results

<p align="center">
<img src="figures/RA_results.png" alt="drawing" width="700"/>
<img src="figures/Sjogren_results.png" alt="drawing" width="700"/>
</p>

<p align="center">
<img src="figures/heatmap_visualisation.png" alt="drawing" width="800"/>
</p>

## How to run the code (WIP)

### Basic Usage

1. Set up your configuration file (e.g., RA_config.yaml)
2. Run the complete pipeline:

```bash
python main.py --config RA_config.yaml --preprocess --train --test --visualise
```
