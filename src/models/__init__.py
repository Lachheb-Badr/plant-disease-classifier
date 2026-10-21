from src.models.base_model import BaseModel
from src.models.custom_cnn import CustomCNN
from src.models.pretrained import PretrainedClassifier


def get_model(config: dict) -> BaseModel:
    """Build the model selected by `model.name` in the config."""
    mcfg = config["model"]
    num_classes = config["data"]["num_classes"]
    name = mcfg["name"]

    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout=mcfg["dropout"])
    if name in PretrainedClassifier.SUPPORTED:
        return PretrainedClassifier(
            name=name,
            num_classes=num_classes,
            pretrained=mcfg["pretrained"],
            dropout=mcfg["dropout"],
            unfreeze_blocks=mcfg.get("unfreeze_blocks"),
        )
    raise ValueError(f"Unknown model name: {name}")
