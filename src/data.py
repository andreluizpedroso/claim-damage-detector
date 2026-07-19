"""Download e pipeline de dados para o dataset real de danos em veículos.

Dataset: "Car Damage Detection" (anujms) no Kaggle.
https://www.kaggle.com/datasets/anujms/car-damage-detection

Estrutura esperada após o download (pasta `data1a`):
    training/00-damage/*.jpg
    training/01-whole/*.jpg
    validation/00-damage/*.jpg
    validation/01-whole/*.jpg
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

DATASET_SLUG = "anujms/car-damage-detection"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def download_dataset() -> Path:
    """Baixa o dataset via kagglehub (requer conta Kaggle configurada) e retorna o
    diretório onde ele foi salvo em cache pelo kagglehub."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_SLUG))


def find_dataset_root(search_dir: Path) -> Path:
    """Procura recursivamente a pasta que contém `training/` e `validation/`."""
    if (search_dir / "training").is_dir() and (search_dir / "validation").is_dir():
        return search_dir
    for candidate in search_dir.rglob("training"):
        if (candidate.parent / "validation").is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"Não foi possível encontrar 'training/' e 'validation/' dentro de {search_dir}. "
        "Baixe o dataset manualmente em https://www.kaggle.com/datasets/anujms/car-damage-detection "
        f"e extraia-o em {DATA_DIR}."
    )


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    if dataset_root is not None:
        return find_dataset_root(Path(dataset_root))
    if DATA_DIR.exists():
        try:
            return find_dataset_root(DATA_DIR)
        except FileNotFoundError:
            pass
    return find_dataset_root(download_dataset())


def _augmentation_layers() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        # `value_range` precisa bater com a escala já normalizada ([0, 1]) das
        # imagens de entrada; o default de RandomBrightness é (0, 255), o que
        # produzia pixels fora de [0, 1] sem nenhum clipping visível no shape.
        tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
    ])


def load_classifier_dataset(
    dataset_root: str | Path | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Carrega os conjuntos de treino/validação para o classificador dano-vs-íntegro.

    As imagens são normalizadas para [0, 1] e o treino recebe augmentation leve.
    Retorna (train_ds, val_ds, class_names).
    """
    root = resolve_dataset_root(dataset_root)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / "training",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root / "validation",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )
    class_names = train_ds.class_names

    augment = _augmentation_layers()
    train_ds = train_ds.map(
        lambda x, y: (augment(x / 255.0, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(
        lambda x, y: (x / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def class_sample_counts(dataset_root: str | Path | None = None, class_names: list[str] | None = None) -> dict[str, int]:
    """Conta quantos arquivos existem em `training/<classe>` para cada classe.

    Usado para calcular pesos de classe (class_weight) no treino do classificador,
    já que o dataset não é perfeitamente balanceado entre 00-damage/01-whole.
    """
    root = resolve_dataset_root(dataset_root)
    train_dir = root / "training"
    names = class_names or sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    return {name: sum(1 for _ in (train_dir / name).glob("*")) for name in names}


def load_gan_images(
    dataset_root: str | Path | None = None,
    class_name: str | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Carrega imagens reais de treino (sem rótulo) normalizadas para [-1, 1],
    o formato esperado pelo gerador/discriminador em `src/gan.py`.

    Se `class_name` for informado ("00-damage" ou "01-whole"), carrega apenas
    imagens dessa classe -- útil para treinar uma GAN focada em uma classe
    minoritária para aumento de dados.
    """
    root = resolve_dataset_root(dataset_root)
    directory = root / "training" / class_name if class_name else root / "training"

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    ds = ds.map(
        lambda x: (x / 127.5) - 1.0,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    return ds
