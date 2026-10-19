from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MnistTensorDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Dataset for preprocessed MNIST .pt files created by scripts/preprocess_mnist.py.
    """

    def __init__(self, split_file: str | Path) -> None:
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"MNIST split file not found: {split_path}")

        payload = torch.load(split_path, map_location="cpu")
        if "images" not in payload or "labels" not in payload:
            raise ValueError(
                f"Invalid split file format in {split_path}. Expected keys: images, labels."
            )

        self.images: Tensor = payload["images"].float()
        self.labels: Tensor = payload["labels"].long()

        if self.images.ndim != 4 or self.images.shape[1:] != (1, 28, 28):
            raise ValueError(
                f"Expected images shape [N, 1, 28, 28], got {tuple(self.images.shape)}"
            )
        if self.labels.ndim != 1 or self.labels.shape[0] != self.images.shape[0]:
            raise ValueError(
                "Expected labels shape [N] and matching first dimension with images."
            )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.images[idx], self.labels[idx]
