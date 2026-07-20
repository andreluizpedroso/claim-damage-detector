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
    """Reproduz manualmente o mesmo pré-processamento do pipeline de treino
    (src/data.py: resize + normalização [0,1]), já que aqui a imagem vem de
    um arquivo avulso e não passa pelo `tf.data.Dataset`."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
    # OpenCV lê em BGR por padrão; o modelo foi treinado com imagens em RGB
    # (via tf.keras.utils.image_dataset_from_directory), então é preciso converter.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    # O modelo espera um batch (N, H, W, C); uma imagem sozinha vira um "batch de 1".
    return np.expand_dims(image, axis=0)


def predict(image_path: str, model: tf.keras.Model | None = None) -> tuple[str, float]:
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    image = load_image(image_path)
    # A saída é um único neurônio sigmoid (índice [0][0] extrai o valor escalar
    # do batch de 1 imagem): quanto mais perto de 1, mais "íntegro" segundo o modelo.
    probability = float(model.predict(image, verbose=0)[0][0])

    # Threshold 0.5 é a escolha padrão/ingênua para decidir a classe a partir da
    # probabilidade. Em produção, esse valor poderia ser ajustado (ex. exigir >0.7
    # para "íntegro") para reduzir falsos negativos na classe de dano, que é o
    # erro mais caro nesse caso de uso (veja DAMAGE_CLASS_BOOST em train_classifier.py).
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
