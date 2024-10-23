
# ImmunoHisto Benchmark

We assess performance of histopathology foundation models trained on H&E cancer 
dataset on OOD Immunohistochemistry (IHC) autoimmune datasets. 

## Models benchmarked

![models_benchmarked.png](models_benchmarked.png)

## Results

<p align="center">
<img src="RA_results.png" alt="drawing" width="500"/>
<img src="Sjogren_results.png" alt="drawing" width="500"/>
</p>

![heatmap_visualisation.png](heatmap_visualisation.png)

## How to run the code (WIP)

### Basic Usage

1. Set up your configuration file (e.g., RA_config.yaml)
2. Run the complete pipeline:

```bash
python main.py --config RA_config.yaml --preprocess --train --test --visualise
```