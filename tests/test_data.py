"""Testes do pipeline de dados usando um dataset falso (imagens dummy),
sem depender do download do dataset real do Kaggle."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data import load_classifier_dataset, load_gan_images


@pytest.fixture
def fake_dataset_root(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("training", "validation"):
        for class_name in ("00-damage", "01-whole"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(4):
                pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"img_{i}.jpg")
    return tmp_path


def test_load_classifier_dataset_shapes_and_range(fake_dataset_root):
    train_ds, val_ds, class_names = load_classifier_dataset(
        dataset_root=fake_dataset_root, image_size=(16, 16), batch_size=2
    )
    assert class_names == ["00-damage", "01-whole"]

    images, labels = next(iter(train_ds))
    assert images.shape[1:] == (16, 16, 3)
    assert labels.shape[1:] == (1,)
    assert float(images.numpy().min()) >= 0.0
    assert float(images.numpy().max()) <= 1.0

    val_images, _ = next(iter(val_ds))
    assert val_images.shape[1:] == (16, 16, 3)


def test_load_gan_images_range_and_class_filter(fake_dataset_root):
    ds = load_gan_images(
        dataset_root=fake_dataset_root, class_name="00-damage", image_size=(16, 16), batch_size=2
    )
    batch = next(iter(ds))
    assert batch.shape[1:] == (16, 16, 3)
    assert float(batch.numpy().min()) >= -1.0
    assert float(batch.numpy().max()) <= 1.0
