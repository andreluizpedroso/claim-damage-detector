"""Treina o classificador dano-vs-íntegro e reporta métricas de avaliação.

Uso:
    python -m src.train_classifier --epochs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
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
    """Calcula um peso por classe para usar no `class_weight` do Keras.

    A fórmula `total / (n_classes * contagem_da_classe)` é o balanceamento
    "clássico" de scikit-learn/Keras: classes com menos exemplos recebem peso
    maior, para que o erro do modelo nelas pese tanto quanto nas classes
    maiores durante o treino (sem isso, um dataset desbalanceado tende a
    fazer o modelo "preguiçosamente" favorecer a classe majoritária).
    Depois, multiplicamos a classe de dano por DAMAGE_CLASS_BOOST -- um ajuste
    de negócio por cima do balanceamento estatístico.
    """
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


def evaluate(model, val_ds, class_names: list[str], plots_dir: Path | None) -> None:
    """Avalia no conjunto de validação (nunca visto durante o `model.fit`) e
    imprime métricas por classe -- accuracy sozinha esconde desequilíbrios
    (ex. um modelo que sempre prevê "íntegro" teria ~50% de accuracy num
    dataset balanceado, mas recall 0 na classe de dano)."""
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_true.extend(labels.numpy().ravel().tolist())
        # Mesmo threshold 0.5 usado em predict.py -- ver comentário lá sobre
        # por que esse valor poderia ser ajustado em produção.
        y_pred.extend((probs >= 0.5).astype(int).tolist())

    print("\nRelatório de classificação (conjunto de validação):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusão:")
    print(cm)

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de confusão (validação)")
        fig.tight_layout()
        fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)


def save_history_plot(history, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy por época")
    axes[0].set_xlabel("época")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss por época")
    axes[1].set_xlabel("época")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "training_history.png", dpi=150)
    plt.close(fig)


def train(epochs: int, batch_size: int, plots_dir: Path | None) -> None:
    train_ds, val_ds, class_names = load_classifier_dataset(batch_size=batch_size)
    print(f"Classes: {class_names}")

    class_weight = compute_class_weight(class_names)
    print(f"Pesos de classe: {dict(zip(class_names, class_weight.values()))}")

    model = build_classifier()
    # `restore_best_weights=True` é o detalhe mais importante aqui: sem ele,
    # o Keras usaria os pesos da ÚLTIMA época treinada, mesmo que uma época
    # anterior tivesse tido val_accuracy melhor (isso já aconteceu de fato
    # numa rodada de treino deste projeto -- a época 4 tinha 77% de
    # val_accuracy e a 15ª, salva por último, tinha caído para 71%).
    # `patience=5` para o treino se val_accuracy não melhorar por 5 épocas
    # seguidas, economizando tempo sem precisar adivinhar o número certo de
    # épocas de antemão.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[early_stopping],
    )

    evaluate(model, val_ds, class_names, plots_dir)
    if plots_dir is not None:
        save_history_plot(history, plots_dir)
        print(f"Gráficos salvos em {plots_dir}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "classifier.keras")
    print(f"Modelo salvo em {MODELS_DIR / 'classifier.keras'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Se informado, salva matriz de confusão e curvas de accuracy/loss como PNG nesse diretório.",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    train(args.epochs, args.batch_size, plots_dir)


if __name__ == "__main__":
    main()
