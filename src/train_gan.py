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
from src.gan import LATENT_DIM, build_discriminator, build_generator

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

    d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy()

    seed = tf.random.normal((16, LATENT_DIM))

    # Loop manual com GradientTape em vez do padrão "gan = Model(discriminator(generator))"
    # com discriminator.trainable = False: no Keras 3, alternar `.trainable` no mesmo
    # objeto de discriminador compartilhado desativava também o treino standalone dele
    # (warning "model does not have any trainable weights"), deixando o discriminador
    # congelado em pesos aleatórios e o gerador "vencendo" trivialmente (g_loss -> 0,
    # amostras viravam ruído puro). GradientTape aplica gradientes só nas variáveis
    # certas em cada etapa, sem depender de toggles de estado.
    @tf.function
    def train_discriminator_step(real_batch):
        batch_size_actual = tf.shape(real_batch)[0]
        noise = tf.random.normal((batch_size_actual, LATENT_DIM))
        fake_batch = generator(noise, training=True)

        labels_real = tf.ones((batch_size_actual, 1)) * 0.9  # label smoothing
        labels_fake = tf.zeros((batch_size_actual, 1))

        with tf.GradientTape() as tape:
            pred_real = discriminator(real_batch, training=True)
            pred_fake = discriminator(fake_batch, training=True)
            loss = 0.5 * (bce(labels_real, pred_real) + bce(labels_fake, pred_fake))
        grads = tape.gradient(loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        return loss

    @tf.function
    def train_generator_step(batch_size_actual):
        noise = tf.random.normal((batch_size_actual, LATENT_DIM))
        labels_real = tf.ones((batch_size_actual, 1))

        with tf.GradientTape() as tape:
            fake_batch = generator(noise, training=True)
            pred_fake = discriminator(fake_batch, training=False)
            loss = bce(labels_real, pred_fake)
        grads = tape.gradient(loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return loss

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for real_batch in dataset:
            d_losses.append(float(train_discriminator_step(real_batch)))
            g_losses.append(float(train_generator_step(tf.shape(real_batch)[0])))

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
