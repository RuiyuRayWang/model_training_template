# Model Training Template

Reusable PyTorch template for local model development and experiments.

## Directory Structure

```
model_training_template/
├── checkpoints/        # Saved training checkpoints
├── configs/            # Optional config files
├── data/               # Raw downloaded data
├── datasets/           # Dataset loaders + processed data outputs
├── models/             # Model definitions
├── notebooks/          # Experiment notebooks
├── scripts/            # Utility / preprocessing / training scripts
├── utils/              # Shared helper functions
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision

Install dependencies:

```bash
pip install -r requirements.txt
```

## MNIST Model Template

A ready-to-use CNN model is available at [mnist_cnn.py](/home/ray/model_training_template/models/mnist_cnn.py).

```python
from models import MnistCNN

model = MnistCNN(num_classes=10)
```

## MNIST Local Download + Preprocessing

### Option A: Download raw MNIST only

```bash
python -c "from torchvision.datasets import MNIST; MNIST(root='data/mnist', train=True, download=True); MNIST(root='data/mnist', train=False, download=True)"
```

### Option B (recommended): Download + preprocess into training-ready tensors

Use [preprocess_mnist.py](/home/ray/model_training_template/scripts/preprocess_mnist.py):

```bash
python scripts/preprocess_mnist.py --raw-dir data/mnist --out-dir datasets/mnist --val-size 5000 --seed 42
```

This generates:
- `datasets/mnist/train.pt`
- `datasets/mnist/val.pt`
- `datasets/mnist/test.pt`
- `datasets/mnist/metadata.json`

Each `.pt` split file contains:
- `images`: `float32`, shape `[N, 1, 28, 28]`, normalized to `[0, 1]`
- `labels`: `int64`, shape `[N]`

## Loading Preprocessed MNIST Splits

Use [mnist_tensor_dataset.py](/home/ray/model_training_template/datasets/mnist_tensor_dataset.py):

```python
from torch.utils.data import DataLoader
from datasets import MnistTensorDataset

train_ds = MnistTensorDataset("datasets/mnist/train.pt")
val_ds = MnistTensorDataset("datasets/mnist/val.pt")
test_ds = MnistTensorDataset("datasets/mnist/test.pt")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
```
