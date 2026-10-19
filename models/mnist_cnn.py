import torch
from torch import nn


class MnistCNN(nn.Module):
    """
    Template CNN for MNIST classification.

    Input:
        - Tensor shape: (batch_size, 1, 28, 28)
    Output:
        - Logits shape: (batch_size, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.25,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits
