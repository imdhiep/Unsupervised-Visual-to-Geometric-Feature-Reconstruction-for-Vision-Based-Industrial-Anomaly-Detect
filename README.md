# Unsupervised geometric-guided industrial anomaly detection

## Paper
Our paper is now publicly available. For more details, please visit: [IEEE Xplore](https://ieeexplore.ieee.org/document/10820339).

![Overview](docs/overview.png)

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Notes](#notes)

## Installation
You should install all the necessary dependencies in the `./requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── .vscode/                  # VS Code configuration
├── .devkit/                  # Development tools
├── utils/                    # Utility functions
│   ├── general_utils.py      # General utility functions
│   ├── mvtec3d_utils.py      # MVTec 3D dataset utilities
│   └── pointnet2_utils.py    # PointNet2 utilities
├── processing/               # Data processing scripts
│   ├── aggregate_results.py  # Results aggregation
│   └── preprocess_mvtec.py   # MVTec dataset preprocessing
├── models/                   # Model implementations
│   ├── ad_models.py         # Anomaly detection models
│   ├── dataset.py           # Dataset classes
│   ├── feature_transfer_nets.py  # Feature transfer networks
│   └── features.py          # Feature extraction
├── docs/                     # Documentation and images
│   ├── overview.png         # Project overview image
│   └── result.png           # Result visualization
├── training.py              # Training script
├── inference.py             # Inference script
├── train.sh                 # Training shell script
├── eval.sh                  # Evaluation shell script
└── requirements.txt         # Project dependencies
```

## Datasets
### MVTec 3D-AD
- Download from: [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)
- Preprocess using: `python processing/preprocess_mvtec.py`
- Supported classes:
  - bagel
  - cable_gland
  - carrot
  - cookie
  - dowel
  - foam
  - peach
  - potato
  - rope
  - tire

### Synthetic Dataset
- Download from: [Synthetic Dataset (Simulation)](https://github.com/synthetic-dataset/simulation)
- Supported classes:
  - CandyCane
  - ChocolateCookie
  - ChocolatePraline
  - Confetto
  - GummyBear
  - HazelnutTruffle
  - LicoriceSandwich
  - Lollipop
  - Marshmallow
  - PeppermintCandy

## Training
To train the network, use the example in `train.sh` or run directly:

```bash
python training.py \
    --dataset_path ./datasets/mvtec3d \
    --checkpoint_savepath ./checkpoints/checkpoints_mvtec \
    --class_name bagel \
    --epochs_no 50 \
    --batch_size 4
```

### Training Parameters
- `--dataset_path`: Path to the root directory of the dataset
- `--checkpoint_savepath`: Path to save checkpoints (e.g., `checkpoints/checkpoints_mvtec`)
- `--class_name`: Class to train on (see supported classes above)
- `--epochs_no`: Number of training epochs
- `--batch_size`: Batch size for training

## Inference
To run inference and generate anomaly maps:

```bash
python inference.py \
    --dataset_path ./datasets/mvtec3d \
    --checkpoint_folder ./checkpoints/checkpoints_mvtec \
    --class_name bagel \
    --epochs_no 50 \
    --batch_size 4 \
    --qualitative_folder ./results/qualitative \
    --quantitative_folder ./results/quantitative \
    --visualize_plot True \
    --produce_qualitatives True
```

### Inference Parameters
- `--dataset_path`: Path to the dataset
- `--checkpoint_folder`: Path to trained checkpoints
- `--class_name`: Class to evaluate
- `--epochs_no`: Number of epochs used in training
- `--batch_size`: Batch size for inference
- `--qualitative_folder`: Folder to save anomaly maps
- `--quantitative_folder`: Folder to save metrics
- `--visualize_plot`: Flag to visualize results during inference
- `--produce_qualitatives`: Flag to save qualitative results

## Results
![Result](docs/result.png)

## Notes
- The code uses `wandb` for experiment tracking during training
- To disable wandb, set `mode = disabled` in `training.py`
- For optimal performance:
  - Use GPU for training and inference
  - Recommended batch size: 4-16
  - Training time varies by class and dataset size
  - Checkpoint regularly to save progress
```
