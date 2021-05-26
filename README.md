# KECRS


**K**nowledge-**E**nriched **C**onversational **R**ecommendation  **S**ystem.<br>

## Prerequisites
- Python 3.6
- PyTorch 1.4.0
- Torch-Geometric 1.4.2

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/Anonymous-Ahhh/KECRS.git
cd KECRS/parlai/task/crs/
```

### Dataset
All the data are in * ./KECRS/data/crs/ *folder
- **ReDial** dataset
- The Movie Domain Knowledge Graph, TMDKG

### Training

To train the recommender part, run:

```bash
python train_kecrs.py
```

To train the dialog part, run:

```bash
python train_transformer_rec.py
```

### Logging

TensorBoard logs and models will be saved in `saved/` folder.

### Evaluation

All results on testing set will be shown after training.

TODO

If you have difficulties to get things working in the above steps, please let us know.
