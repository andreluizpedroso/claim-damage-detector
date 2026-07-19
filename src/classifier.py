"""Classificador binário dano-vs-íntegro via transfer learning (MobileNetV2).

Este é o modelo que de fato deve ser usado para inferência (`src/predict.py`) --
o discriminador da GAN em `src/gan.py` serve para julgar imagens reais vs.
sintéticas durante o treino adversarial, não é um classificador de negócio.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SHAPE = (128, 128, 3)


def build_classifier(image_shape: tuple[int, int, int] = IMAGE_SHAPE, fine_tune_at: int = 140) -> keras.Model:
    """`fine_tune_at=140` mantém apenas as ~14 últimas camadas do MobileNetV2
    treináveis. Um `fine_tune_at` menor (mais camadas destravadas) desestabiliza
    o treino nesse dataset pequeno (~1150 imagens): o val_accuracy oscila entre
    épocas em vez de melhorar de forma consistente."""
    base_model = keras.applications.MobileNetV2(
        input_shape=image_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = keras.Input(shape=image_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="damage_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
