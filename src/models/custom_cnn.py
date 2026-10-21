"""Small CNN built from scratch: the baseline model."""
import torch
import torch.nn as nn

from src.models.base_model import BaseModel


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(BaseModel):
    """4 conv blocks (16 -> 32 -> 64 -> 128) + global average pooling + linear head.

    Global average pooling makes the network independent of the input size,
    so the non-square 224x336 input works without any change.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3, channels=(16, 32, 64, 128)):
        super().__init__(num_classes)
        layers, in_ch = [], 3
        for out_ch in channels:
            layers.append(conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))