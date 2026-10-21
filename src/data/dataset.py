"""PyTorch Dataset over a Split/Class folder structure."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PlantDiseaseDataset(Dataset):
    """Loads images from `root_dir/<class_name>/*.jpg`.

    Class indices follow the order of `classes` (taken from the config), so the
    label mapping is the same for every split.
    """

    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = Path(root_dir)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))
        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
