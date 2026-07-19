"""DCGAN convolucional para gerar imagens sintéticas de carros (128x128x3).

Diferente da versão original do notebook, o discriminador é treinado contra
imagens reais do dataset (veja `src/data.py`), não ruído aleatório.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

LATENT_DIM = 100
IMAGE_SHAPE = (128, 128, 3)


def build_generator(latent_dim: int = LATENT_DIM) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 256),
        layers.Reshape((8, 8, 256)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(128, 4, strides=2, padding="same"),  # 16x16
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, 4, strides=2, padding="same"),  # 32x32
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(32, 4, strides=2, padding="same"),  # 64x64
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh"),  # 128x128
    ], name="generator")
    return model


def build_discriminator(image_shape: tuple[int, int, int] = IMAGE_SHAPE) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=image_shape),
        layers.Conv2D(32, 4, strides=2, padding="same"),  # 64x64
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(64, 4, strides=2, padding="same"),  # 32x32
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 4, strides=2, padding="same"),  # 16x16
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ], name="discriminator")
    return model


def build_gan(generator: keras.Model, discriminator: keras.Model, latent_dim: int = LATENT_DIM) -> keras.Model:
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    return keras.Model(gan_input, gan_output, name="gan")
