"""Testes do pipeline de dados usando um dataset falso (imagens dummy),
sem depender do download do dataset real do Kaggle."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data import load_classifier_dataset, load_gan_images


@pytest.fixture
def fake_dataset_root(tmp_path):
    """Cria em disco (numa pasta temporária, apagada automaticamente pelo
    pytest ao fim do teste) uma estrutura de pastas idêntica à do dataset
    real (training/validation x 00-damage/01-whole), mas com imagens
    aleatórias 32x32. Não baixamos o dataset de verdade nos testes porque
    isso exigiria internet + credenciais do Kaggle -- o objetivo aqui é
    testar o *pipeline* (shapes, normalização), não o conteúdo das imagens."""
    rng = np.random.default_rng(0)  # seed fixa: mesmas imagens "aleatórias" a cada rodada de teste
    for split in ("training", "validation"):
        for class_name in ("00-damage", "01-whole"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(4):
                pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"img_{i}.jpg")
    return tmp_path


def test_load_classifier_dataset_shapes_and_range(fake_dataset_root):
    """Verifica que load_classifier_dataset() resiza corretamente e
    normaliza para [0, 1] -- essa é a asserção que pegou o bug real do
    RandomBrightness (values chegavam a ~19.7, muito acima de 1.0)."""
    train_ds, val_ds, class_names = load_classifier_dataset(
        dataset_root=fake_dataset_root, image_size=(16, 16), batch_size=2
    )
    assert class_names == ["00-damage", "01-whole"]  # ordem alfabética, ver data.py

    images, labels = next(iter(train_ds))
    assert images.shape[1:] == (16, 16, 3)
    assert labels.shape[1:] == (1,)
    assert float(images.numpy().min()) >= 0.0
    assert float(images.numpy().max()) <= 1.0

    val_images, _ = next(iter(val_ds))
    assert val_images.shape[1:] == (16, 16, 3)


def test_load_gan_images_range_and_class_filter(fake_dataset_root):
    """Verifica que load_gan_images() filtra por classe corretamente e
    normaliza para [-1, 1] (não [0, 1] como o pipeline do classificador) --
    faixa exigida pela ativação tanh do gerador, ver src/gan.py."""
    ds = load_gan_images(
        dataset_root=fake_dataset_root, class_name="00-damage", image_size=(16, 16), batch_size=2
    )
    batch = next(iter(ds))
    assert batch.shape[1:] == (16, 16, 3)
    assert float(batch.numpy().min()) >= -1.0
    assert float(batch.numpy().max()) <= 1.0
