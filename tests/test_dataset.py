from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.data.dataloader import get_dataloaders  # noqa: E402
from src.data.dataset import PlantDiseaseDataset  # noqa: E402
from src.data.transforms import get_eval_transforms, get_train_transforms  # noqa: E402


def train_dir(cfg):
    return Path(cfg["data"]["processed_dir"]) / cfg["data"]["splits"]["train"]


def test_dataset_length_and_labels(fake_data):
    ds = PlantDiseaseDataset(train_dir(fake_data), fake_data["data"]["classes"])
    assert len(ds) == 12
    assert {label for _, label in ds.samples} == {0, 1, 2}


def test_missing_class_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        PlantDiseaseDataset(tmp_path, ["Healthy"])


def test_dataloaders_shapes(fake_data):
    loaders = get_dataloaders(fake_data)
    assert set(loaders) == {"train", "val", "test"}
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, 3, 64, 96) and x.dtype == torch.float32
    assert 0 <= y.min() and y.max() <= 2
    assert len(loaders["val"].dataset) == 12


def test_eval_transform_is_deterministic(fake_data):
    ds = PlantDiseaseDataset(train_dir(fake_data), fake_data["data"]["classes"], get_eval_transforms(fake_data))
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_train_transform_output_shape(fake_data):
    ds = PlantDiseaseDataset(train_dir(fake_data), fake_data["data"]["classes"], get_train_transforms(fake_data))
    x, _ = ds[0]
    assert x.shape == (3, 64, 96)
