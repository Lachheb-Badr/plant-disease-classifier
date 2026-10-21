"""Load a trained model from a checkpoint saved by the Trainer."""
import torch

from src.models import get_model


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode, config stored in the checkpoint)."""
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    config["model"]["pretrained"] = False  # the weights come from the checkpoint
    model = get_model(config)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), config
