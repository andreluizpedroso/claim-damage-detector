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
    """Recebe um vetor de ruído aleatório (o "código latente", 100 números)
    e o transforma progressivamente numa imagem 128x128x3. Cada
    Conv2DTranspose dobra a resolução (8→16→32→64→128), o oposto do que um
    classificador normal faz (que via de regra reduz resolução a cada camada)."""
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        # Projeta o vetor 1D de ruído num "mapa" 3D pequeno (8x8x256), que
        # depois vai sendo espichado espacialmente pelas camadas seguintes.
        layers.Dense(8 * 8 * 256),
        layers.Reshape((8, 8, 256)),
        layers.BatchNormalization(),  # estabiliza o treino, comum em GANs
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

        # Ativação tanh na última camada: força a saída para o intervalo
        # [-1, 1], que é a mesma escala em que as imagens reais são
        # normalizadas em load_gan_images() (src/data.py) -- sem isso, o
        # discriminador estaria comparando imagens em escalas diferentes.
        layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh"),  # 128x128
    ], name="generator")
    return model


def build_discriminator(image_shape: tuple[int, int, int] = IMAGE_SHAPE) -> keras.Model:
    """Um classificador binário comum (imagem real vs. gerada) -- estrutura
    quase espelhada à do gerador: em vez de aumentar a resolução, cada
    Conv2D a reduz pela metade (128→64→32→16) até virar um vetor único."""
    model = keras.Sequential([
        layers.Input(shape=image_shape),
        layers.Conv2D(32, 4, strides=2, padding="same"),  # 64x64
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),  # ajuda a evitar que o discriminador aprenda rápido demais

        layers.Conv2D(64, 4, strides=2, padding="same"),  # 32x32
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 4, strides=2, padding="same"),  # 16x16
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        # 1 neurônio + sigmoid = probabilidade de a imagem ser "real" (perto
        # de 1) vs. "gerada pela GAN" (perto de 0).
        layers.Dense(1, activation="sigmoid"),
    ], name="discriminator")
    return model

