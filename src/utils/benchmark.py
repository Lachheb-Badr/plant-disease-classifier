"""Quick CPU benchmark: is training limited by data loading or by the model?

Usage (from the project root):
    python -m src.utils.benchmark --model custom_cnn
    python -m src.utils.benchmark --model custom_cnn --image-size 128 192
    python -m src.utils.benchmark --model efficientnet_b0 --image-size 224 336
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from src.data.dataloader import get_dataloaders
from src.models import get_model
from src.utils.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model")
    p.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"))
    p.add_argument("--batches", type=int, default=5)
    args = p.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.image_size:
        config["data"]["image_size"] = args.image_size
    config["model"]["pretrained"] = False  # weights don't matter for timing

    h, w = config["data"]["image_size"]
    bs = config["dataloader"]["batch_size"]
    print(f"model={config['model']['name']} | input={h}x{w} | batch={bs} | "
          f"cpu cores={os.cpu_count()} | torch threads={torch.get_num_threads()}")

    # 1) data loading alone
    loader = get_dataloaders(config)["train"]
    n_batches = len(loader)
    it = iter(loader)
    next(it)  # warm-up: worker start-up
    t = time.time()
    for _ in range(args.batches):
        next(it)
    data_s = (time.time() - t) / args.batches

    # 2) model forward + backward alone, on random data
    model = get_model(config)
    model.train()
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad])
    x = torch.randn(bs, 3, h, w)
    y = torch.randint(0, config["data"]["num_classes"], (bs,))

    def step():
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    step()  # warm-up
    t = time.time()
    for _ in range(args.batches):
        step()
    model_s = (time.time() - t) / args.batches

    print(f"data loading : {data_s:6.2f} s/batch")
    print(f"model step   : {model_s:6.2f} s/batch")
    print(f"batches/epoch: {n_batches}")
    print(f"estimated epoch: {max(data_s, model_s) * n_batches / 60:.1f} min "
          f"(up to {(data_s + model_s) * n_batches / 60:.1f} min if both compete for the CPU)")
    print("bottleneck   :", "data loading" if data_s > model_s else "model computation")


if __name__ == "__main__":
    main()
