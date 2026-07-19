"""Treina a DCGAN em imagens reais do dataset e salva amostras geradas.

Uso:
    python -m src.train_gan --epochs 50 --class-name 00-damage
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data import load_gan_images
from src.gan import LATENT_DIM, build_discriminator, build_gan, build_generator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "outputs" / "samples"
MODELS_DIR = PROJECT_ROOT / "models"


def save_sample_grid(generator: tf.keras.Model, epoch: int, seed: tf.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = generator(seed, training=False).numpy()
    generated = (generated + 1.0) / 2.0  # volta para [0, 1]

    n = generated.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(np.array(axes).reshape(-1)):
        ax.axis("off")
        if i < n:
            ax.imshow(np.clip(generated[i], 0, 1))
    fig.tight_layout()
    fig.savefig(out_dir / f"epoch_{epoch:04d}.png")
    plt.close(fig)


def train(epochs: int, batch_size: int, class_name: str | None, sample_every: int) -> None:
    dataset = load_gan_images(class_name=class_name, batch_size=batch_size)

    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5), loss="binary_crossentropy")

    seed = tf.random.normal((16, LATENT_DIM))

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for real_batch in dataset:
            batch_size_actual = real_batch.shape[0]
            noise = tf.random.normal((batch_size_actual, LATENT_DIM))
            fake_batch = generator(noise, training=True)

            labels_real = tf.ones((batch_size_actual, 1)) * 0.9  # label smoothing
            labels_fake = tf.zeros((batch_size_actual, 1))

            d_loss_real = discriminator.train_on_batch(real_batch, labels_real)
            d_loss_fake = discriminator.train_on_batch(fake_batch, labels_fake)
            d_losses.append(0.5 * (d_loss_real[0] + d_loss_fake[0]))

            noise = tf.random.normal((batch_size_actual, LATENT_DIM))
            g_loss = gan.train_on_batch(noise, tf.ones((batch_size_actual, 1)))
            g_losses.append(g_loss)

        print(f"epoch {epoch}/{epochs} - d_loss={np.mean(d_losses):.4f} g_loss={np.mean(g_losses):.4f}")

        if epoch % sample_every == 0 or epoch == epochs:
            save_sample_grid(generator, epoch, seed, SAMPLES_DIR)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    generator.save(MODELS_DIR / "generator.keras")
    discriminator.save(MODELS_DIR / "discriminator.keras")
    print(f"Modelos salvos em {MODELS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        choices=["00-damage", "01-whole"],
        help="Treina a GAN apenas em uma classe (para gerar amostras sintéticas dessa classe).",
    )
    parser.add_argument("--sample-every", type=int, default=5)
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.class_name, args.sample_every)


if __name__ == "__main__":
    main()
