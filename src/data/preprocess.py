"""One-time preprocessing: downscale the huge raw images.

Raw images are ~4000x2700. Decoding them every epoch would make training very
slow, so we resize them once to `data.preprocess_size` and store them in
`data.processed_dir`, keeping the Split/Class folder structure.

Usage (from the project root):
    python -m src.data.preprocess --config configs/config.yaml
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from src.utils.config import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(src: Path, dst: Path, size_hw: tuple[int, int]) -> None:
    """Resize one image to (height, width) and save it as a JPEG."""
    height, width = size_hw
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)  # respect phone-camera orientation
        im = im.convert("RGB").resize((width, height), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=95)


def run(config: dict, overwrite: bool = False) -> None:
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    out_dir = Path(data_cfg["processed_dir"])
    size_hw = tuple(data_cfg["preprocess_size"])

    done = skipped = 0
    for split_dir_name in data_cfg["splits"].values():
        for cls in data_cfg["classes"]:
            src_dir = raw_dir / split_dir_name / cls
            if not src_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {src_dir}")
            files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            for src in tqdm(files, desc=f"{split_dir_name}/{cls}"):
                dst = out_dir / split_dir_name / cls / f"{src.stem}.jpg"
                if dst.exists() and not overwrite:
                    skipped += 1
                    continue
                preprocess_image(src, dst, size_hw)
                done += 1
    print(f"Done: {done} images written, {skipped} skipped (already processed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), overwrite=args.overwrite)
