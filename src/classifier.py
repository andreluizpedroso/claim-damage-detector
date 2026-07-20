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
    # `include_top=False` remove a última camada de classificação da rede
    # original (treinada para as 1000 classes do ImageNet); vamos colocar a
    # nossa própria camada de saída (binária) no lugar. `weights="imagenet"`
    # carrega os pesos já treinados -- é isso que torna esse "transfer learning":
    # a rede já sabe reconhecer bordas, texturas e formas antes mesmo de ver
    # uma imagem de carro.
    base_model = keras.applications.MobileNetV2(
        input_shape=image_shape, include_top=False, weights="imagenet"
    )
    # Congela a maior parte da rede pré-treinada e destrava só as últimas
    # camadas para fine-tuning. Isso evita destruir o conhecimento geral já
    # aprendido no ImageNet (que tem ~1,4 milhão de imagens) usando só as
    # ~1150 imagens de treino deste projeto -- ver docstring da função.
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = keras.Input(shape=image_shape)
    # As imagens do pipeline (src/data.py) já vêm normalizadas para [0, 1];
    # multiplicamos de volta por 255 porque o pré-processamento oficial do
    # MobileNetV2 (preprocess_input) espera pixels em [0, 255] e aplica sua
    # própria normalização interna (a mesma usada quando ele foi treinado no
    # ImageNet). Pular esse passo faria o modelo ver os dados numa escala
    # diferente da que aprendeu, prejudicando a qualidade do transfer learning.
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    # training=False aqui mantém as estatísticas de BatchNormalization do
    # ImageNet congeladas mesmo durante o treino -- outra prática padrão de
    # fine-tuning para não desestabilizar o que a rede já aprendeu.
    x = base_model(x, training=False)
    # Reduz o mapa de features (H, W, canais) para um vetor por imagem,
    # tirando a média espacial -- alternativa mais simples e com menos
    # parâmetros do que "achatar" (Flatten) tudo antes da camada densa final.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # regularização: desliga 30% dos neurônios a cada passo de treino
    # 1 neurônio + sigmoid = probabilidade de ser a classe 1 ("01-whole").
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="damage_classifier")
    model.compile(
        # Learning rate baixo (1e-5) de propósito: como estamos fazendo
        # fine-tuning de uma rede já treinada, passos grandes de atualização
        # tendem a "esquecer" rápido demais o que o ImageNet ensinou.
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",  # padrão para classificação binária com saída sigmoid
        metrics=["accuracy"],
    )
    return model
