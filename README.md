# Detecção de Danos em Sinistros com GAN + Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-metrics-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Car%20Damage%20Detection-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/anujms/car-damage-detection)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Projeto de visão computacional que treina um classificador real (transfer learning) para
identificar se uma imagem de sinistro veicular mostra um carro **danificado** ou **íntegro**,
usando dados reais. Uma GAN convolucional (DCGAN) é usada para gerar imagens sintéticas de
carros que podem ser usadas como aumento de dados durante o treino.

## ⚠️ Sobre o escopo (leia antes de usar)

Não existe dataset público rotulado como "fraude em sinistro de seguro" — rótulos de fraude
normalmente vêm de investigação pericial e não são divulgados publicamente por questões de
sigilo e privacidade. Este projeto usa como proxy o dataset real **Car Damage Detection**
(carro danificado vs. íntegro), que é o mais próximo disponível publicamente. Ele é útil para:
- Detectar imagens inconsistentes com o sinistro relatado (ex.: cliente relata dano e envia foto
  de carro íntegro, ou vice-versa) — um sinal real de possível fraude.
- Servir de base para um pipeline de triagem automática que prioriza casos para revisão humana.

Ele **não** classifica fraude diretamente. Para um classificador de fraude de verdade, é
necessário um dataset com rótulos de fraude confirmada, tipicamente proprietário de uma
seguradora.

## 📌 Tecnologias
- Python 3.10+
- TensorFlow / Keras (MobileNetV2 para o classificador, DCGAN para geração de imagens)
- OpenCV, Pillow
- scikit-learn (métricas de avaliação)
- kagglehub (download do dataset)

## 📁 Dataset

[Car Damage Detection (anujms)](https://www.kaggle.com/datasets/anujms/car-damage-detection) —
1610 imagens reais de carros, divididas em `training/` e `validation/`, cada uma com as classes
`00-damage` e `01-whole`.

### Baixar o dataset
1. Crie uma conta no [Kaggle](https://www.kaggle.com) e gere um token de API em
   *Account → Create New API Token* (baixa um `kaggle.json`).
2. Coloque `kaggle.json` em `~/.kaggle/kaggle.json` (Linux/Mac) ou
   `C:\Users\<usuário>\.kaggle\kaggle.json` (Windows).
3. Rode:
   ```bash
   python -c "from src.data import download_dataset; print(download_dataset())"
   ```
   Isso baixa o dataset via `kagglehub` e imprime o caminho local. Os scripts de treino também
   baixam automaticamente na primeira execução se `data/raw/` estiver vazio.

Alternativamente, baixe manualmente pelo site do Kaggle e extraia o conteúdo em `data/raw/`
(precisa conter as pastas `training/` e `validation/`).

## 🛠️ Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar o classificador (o modelo usado em produção)
```bash
python -m src.train_classifier --epochs 25
```
Usa `class_weight` (com peso extra para a classe `00-damage`, já que deixar passar um carro
danificado como "íntegro" é o erro mais caro nesse caso de uso) e `EarlyStopping` com
`restore_best_weights` para manter a melhor época, não a última. Ao final, imprime
`precision`/`recall`/`f1` e a matriz de confusão no conjunto de validação, e salva o modelo em
`models/classifier.keras`. Resultado de referência obtido com o dataset completo: ~84% de
accuracy, 0.97 de precisão e 0.70 de recall na classe `00-damage`.

### 3. (Opcional) Treinar a GAN para gerar imagens sintéticas
```bash
python -m src.train_gan --epochs 50 --class-name 00-damage
```
Salva grades de imagens geradas em `outputs/samples/` a cada N épocas (útil para acompanhar
visualmente a evolução do gerador) e os pesos em `models/generator.keras`.

### 4. Classificar uma imagem
```bash
python -m src.predict caminho/para/imagem.jpg
```

## 🧪 Testes
```bash
pytest tests/
```
Os testes usam imagens sintéticas geradas em tempo de execução (não dependem do dataset real
baixado) e verificam formatos e faixas de normalização do pipeline de dados.

## 📊 Notebooks
- [`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb) — análise
  exploratória do dataset real: balanceamento de classes, dimensões/aspect ratio das imagens,
  brilho médio por classe e amostras visuais.
- [`Fraude Detection Gan.ipynb`](<Fraude Detection Gan.ipynb>) — notebook interativo que chama o
  código de `src/` para treino rápido e visualização (não é a forma recomendada de treinar; use os
  scripts CLI abaixo para treino completo).

## 📂 Estrutura do projeto
```
notebooks/
└── 01_exploratory_analysis.ipynb   # EDA do dataset real
src/
├── data.py              # download + pipeline tf.data (classificador e GAN)
├── gan.py                # DCGAN convolucional (gerador/discriminador)
├── classifier.py          # classificador MobileNetV2 (transfer learning)
├── train_gan.py           # CLI: treina a GAN
├── train_classifier.py    # CLI: treina o classificador e reporta métricas
└── predict.py              # CLI: classifica uma imagem
tests/
└── test_data.py            # testes do pipeline de dados
```

## 📝 Contribuição
Contribuições são bem-vindas! Abra um pull request ou reporte um problema na seção de issues.

## 📜 Licença
MIT License.
