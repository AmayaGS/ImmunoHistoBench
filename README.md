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

## How to run the code

 


### Usage

The code is divided into **preprocessing**, **feature extraction**, **training & validation**, **testing** and **heatmap generation**.

- **Preprocessing**
  - **Segmentation.** A automated tissue segmentation step, using adaptive thresholding to segment tissue areas on the WSIs and extract tissue patches from the image.
  - **Feature extraction.** Each image patch is passed through the feature extractor and embedded into feature vectors. All feature vectors from a given patient are aggregated into a matrix.
- **Training/Validation/Testing** with ABMIL. 
- **Heatmap generation.** 

#### Data Directory Structure

The `patient_labels.csv` and WSIs should be stored in a directory structure as shown below. Store the slides in the `input_directory`. It should contain all the WSIs for each patient, with the naming convention `patientID_staintype.tiff`. The `patient_labels.csv` file should contain the patient IDs and the target labels for the task:

```markdown
--- {Dataset}
    patient_labels.csv
    --- input_directory
            --- patient1_HE.tiff
            --- patient1_CD3.tiff
            --- patient1_CD138.tiff
                .
                .
            --- patientN_HE.tiff
            --- patientN_CD138.tiff
```

#### Config file

All the system arguments and execution flags are defined in the `{Dataset}_config.yaml` file. All the arguments can be modified there for a given run. You can also modify any of these arguments via the command line. 

You should first modify the paths to point towards your input and output folders:

```yaml
paths:
  input_directory: "/path/to/input/slides"
  output_directory: "/path/to/output/folder"
  embedding_weights: "path/to/embedding/weights"
  path_to_patches: "path/to/extracted/patches" # this is for heatmap generation. 
```

You should also modify the parsing and label configs to suit your dataset:

```yaml
# Parsing configurations for 
parsing:
  patient_ID: 'img.split("_")[0]' # "Patient123_stain" -> Patient123
  stain: 'img.split("_")[1]' # "Patient123_stain" -> stain
  stain_types: {'NA': 0, 'H&E': 1, 'CD68': 2, 'CD138': 3, 'CD20': 4} # RA stain types


# Label/split configurations
labels:
  label: 'label'
  label_dict: {'0': 'Pauci-Immune', '1': 'Lymphoid/Myeloid'} # RA subtypes label names
  n_classes: 2
  patient_id: 'Patient_ID'
```

The stain_types dictionary maps the stain types in your dataset to numeric coding. Change the 'label' category to the column name in you `patient_label.csv` file, as well as the patient_id column name.

### Preprocessing

Preprocessing can be run using the following command, where `embedding_net` corresponds to the Feature Extractor model used (for example, CTransPath, UNI, BiOptimus, ... see below for more details):

```bash
python main.py --preprocess --embedding_net 'UNI' --input_directory path/to/slides --directory path/to/output --dataset_name dataset_name
```

`--preprocess` will create 4 new folders: output, dictionaries, masks, contours.

- `masks` contains all the downsampled binary masks obtained during tissue segmentation. 

- `Contours` contain downsampled WSIs with mask contours drawn on thumbnails of the WSIs as a sanity check. You can easily check you're segmenting the right thing and that there's no issues with the WSIs themselves.

- `output` contains the patches folder, containing all the extracted patches, as well as the `extracted_patches.csv` file which contains all the patient_IDs, filenames, coordinates and locations on disk of the patches extracted during the tissue segmentation step.  

- `dictionaries` contains pickled dictionaries of the embedded feature vectors.

Alternatively, each step can be run separately (if you already have binary masks or image patches for example) using the following commands:

```bash
python main.py --segmentation # tissue segmentation and patching
python main.py --embedding # Feature extraction
```

### Feature Extraction

Here, you can choose which feature extractor model you want to use. We currently support the following feature extractors:

- ImageNet pretrained:
  - VGG16
  - ResNet18
  - ResNet50
  - ConvNext
  - ViT_embedding (HuggingFace)
- TCGA pretrained:
  - ssl_resnet18 (Github)
  - ssl_resnet50 (GitHub)
  - CTransPath (GitHub)
  - Lunit (GitHub)
- Proprietary data pretrained:
  - GigaPath (HuggingFace)
  - Phikon (HuggingFace)
  - BiOptimus (HuggingFace)
  - UNI_embedding (HuggingFace)

Set the `embedding_net` parameter to the model you want to run. Check the config file for the model names. You will need to request access on `HuggingFace` for some of these models, set up an access key and download the pretrained weights.

```bash
python main.py --embedding --embedding_net 'UNI' # Feature extraction
```
The embeddings obtained from each feature vectors will be stored in the `dictionaries` folder:

- `dictionaries` contains pickled dictionaries of the embedded feature vectors, named after the model used.

### Training & Testing

#### Model Training

Training is run using the following command:

```bash
python main.py --train --input_directory path/to/slides --directory path/to/output --dataset_name dataset_name
```

The results will be stored in the `output` directory. There you will find training/validation logs for each fold + summary statistics, as well as model weights in the `checkpoints` folder.

Additional training parameters can  modified in the `config.yaml` or set using command-line arguments. For a full list of options, run:

```bash
python main.py --help
```

#### Model Testing

Testing is run on the hold-out test set using the following command:

```bash
python main.py --test --directory path/to/output --dataset_name dataset_name
```

This will test the corresponding model weights on the hold-out test set and store final results in the `output` directory.

### Heatmap Generation

To examine the attention weights obtained after training for each feature extractor model, heatmaps can be generated using the following command:

```bash
python main.py --visualise --directory path/to/output --dataset_name dataset_name --path_to_patches path/to/patches --heatmap_path path/to/save/heatmaps
```

This will generate heatmaps for the test folds using the trained model weights and store them in the `heatmap_path` directory.

--------------

If this code or article were useful to you, please consider citing:

````

@article{ImmunoHisto_GS_2024,
      title={Going Beyond H&E and Oncology: How Do Histopathology Foundation Models Perform for Multi-stain IHC and Immunology?}, 
      author={Amaya Gallagher-Syed and Elena Pontarini and Myles J. Lewis and Michael R. Barnes and Gregory Slabaugh},
      booktitle = {Workshop on Advancements In Medical Foundation Models at NeurIPS 2024},
      year={2024},
      url={https://arxiv.org/abs/2410.21560}, 
}

````

--------------
