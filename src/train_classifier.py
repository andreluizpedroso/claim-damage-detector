"""Treina o classificador dano-vs-íntegro e reporta métricas de avaliação.

Uso:
    python -m src.train_classifier --epochs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.classifier import build_classifier
from src.data import class_sample_counts, load_classifier_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Deixar passar um carro danificado como "íntegro" (falso negativo na classe
# 00-damage) é o erro mais caro no caso de uso de triagem de sinistros, então
# além de balancear pelo tamanho das classes, damos um peso extra a ela.
DAMAGE_CLASS_BOOST = 1.3


def compute_class_weight(class_names: list[str]) -> dict[int, float]:
    counts = class_sample_counts(class_names=class_names)
    total = sum(counts.values())
    weights = {
        i: total / (len(class_names) * counts[name])
        for i, name in enumerate(class_names)
    }
    for i, name in enumerate(class_names):
        if name == "00-damage":
            weights[i] *= DAMAGE_CLASS_BOOST
    return weights


def evaluate(model, val_ds, class_names: list[str]) -> None:
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_true.extend(labels.numpy().ravel().tolist())
        y_pred.extend((probs >= 0.5).astype(int).tolist())

    print("\nRelatório de classificação (conjunto de validação):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Matriz de confusão:")
    print(confusion_matrix(y_true, y_pred))


def train(epochs: int, batch_size: int) -> None:
    train_ds, val_ds, class_names = load_classifier_dataset(batch_size=batch_size)
    print(f"Classes: {class_names}")

    class_weight = compute_class_weight(class_names)
    print(f"Pesos de classe: {dict(zip(class_names, class_weight.values()))}")

    model = build_classifier()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[early_stopping],
    )

    evaluate(model, val_ds, class_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "classifier.keras")
    print(f"Modelo salvo em {MODELS_DIR / 'classifier.keras'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
