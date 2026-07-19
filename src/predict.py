"""Classifica uma imagem de sinistro como possível dano/fraude ou consistente.

Uso:
    python -m src.predict caminho/para/imagem.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.data import IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "classifier.keras"

# `image_dataset_from_directory` ordena as classes alfabeticamente:
# label 0 -> "00-damage", label 1 -> "01-whole" (veja src/train_classifier.py).
CLASS_NAMES = ["00-damage", "01-whole"]


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict(image_path: str, model: tf.keras.Model | None = None) -> tuple[str, float]:
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    image = load_image(image_path)
    probability = float(model.predict(image, verbose=0)[0][0])

    if probability < 0.5:
        label = "Suspeita de dano/fraude detectada"
    else:
        label = "Imagem consistente com sinistro legítimo (carro íntegro)"
    return label, probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=str, help="Caminho para a imagem a classificar")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Modelo não encontrado em {MODEL_PATH}. Rode `python -m src.train_classifier` primeiro."
        )

    label, probability = predict(args.image_path)
    print(f"{label} (confiança de 'íntegro' = {probability:.2%})")


if __name__ == "__main__":
    main()
