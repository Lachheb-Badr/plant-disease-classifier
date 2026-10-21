"""Transfer learning: fine-tune torchvision models with a new classification head."""
import torch
import torch.nn as nn
from torchvision import models

from src.models.base_model import BaseModel


class PretrainedClassifier(BaseModel):
    """MobileNetV3-small, EfficientNet-B0 or ResNet-50 with a new last layer.

    Args:
        name: "mobilenet_v3_small", "efficientnet_b0" or "resnet50"
        pretrained: load ImageNet weights (downloaded on first use)
        dropout: dropout before the new linear layer
        unfreeze_blocks: which part of the network is trained
            None -> everything (full fine-tuning)
            0    -> only the new head (frozen backbone)
            N>0  -> the new head + the last N blocks of the backbone (partial fine-tuning)
            The blocks are `net.features[i]` for MobileNetV3 / EfficientNet
            (EfficientNet-B0 has 9) and layer1..layer4 for ResNet-50.
    """

    SUPPORTED = ("mobilenet_v3_small", "efficientnet_b0", "resnet50")

    def __init__(
        self,
        name: str,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        unfreeze_blocks: int | None = None,
    ):
        super().__init__(num_classes)
        if unfreeze_blocks is not None and unfreeze_blocks < 0:
            raise ValueError("unfreeze_blocks must be None or >= 0")
        self.name = name
        self.unfreeze_blocks = unfreeze_blocks
        self._trainable_blocks = []  # plain list: not registered as sub-modules

        if name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.net = models.efficientnet_b0(weights=weights)
            in_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.net = models.mobilenet_v3_small(weights=weights)
            # classifier = [Linear, Hardswish, Dropout, Linear]: replace dropout + last layer
            in_features = self.net.classifier[3].in_features
            self.net.classifier[2] = nn.Dropout(dropout)
            self.net.classifier[3] = nn.Linear(in_features, num_classes)
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.net = models.resnet50(weights=weights)
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model '{name}'. Choose from {self.SUPPORTED}")

        if unfreeze_blocks is not None:
            for p in self.net.parameters():
                p.requires_grad = False
            blocks = self.backbone_blocks()
            n = min(unfreeze_blocks, len(blocks))
            self._trainable_blocks = blocks[len(blocks) - n:] if n > 0 else []
            for block in self._trainable_blocks:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.head.parameters():
                p.requires_grad = True

    def backbone_blocks(self) -> list:
        if self.name == "resnet50":
            return [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        return list(self.net.features)

    @property
    def head(self) -> nn.Module:
        return self.net.fc if self.name == "resnet50" else self.net.classifier

    def train(self, mode: bool = True):
        """When part of the backbone is frozen, keep the frozen layers (and their
        BatchNorm statistics) in eval mode; only the trainable parts are in train mode."""
        super().train(mode)
        if self.unfreeze_blocks is not None and mode:
            self.net.eval()
            for block in self._trainable_blocks:
                block.train()
            self.head.train()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
