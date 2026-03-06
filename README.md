# Flower Classification with Data Augmentation

Implementação de um modelo de classificação de flores utilizando CNN com TensorFlow/Keras. Durante o treinamento inicial foi observada uma accuracy de aproximadamente 99%, caracterizando overfitting, enquanto a avaliação no conjunto de teste apresentou cerca de 65% de accuracy. Para reduzir o overfitting e melhorar a generalização do modelo, foi implementada a técnica de data augmentation.

## Dataset

O dataset utilizado é **flower_photos**, disponibilizado pelo TensorFlow:

https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

Classes presentes no dataset:

- roses
- daisy
- dandelion
- sunflowers
- tulips

## Pré-processamento

- Download automático do dataset via `tf.keras.utils.get_file`
- Leitura das imagens com `OpenCV`
- Redimensionamento para `180x180`
- Conversão para arrays `NumPy`
- Divisão em treino e teste com `train_test_split`
- Normalização dos pixels dividindo por `255`

## Arquitetura do Modelo

Rede neural convolucional composta por:

- Conv2D (16 filtros) + MaxPooling
- Conv2D (32 filtros) + MaxPooling
- Conv2D (64 filtros) + MaxPooling
- Flatten
- Dense (128 neurônios)
- Dense (5 neurônios, softmax)

Função de perda: `sparse_categorical_crossentropy`  
Otimizador: `Adam`

## Problema Identificado

Durante o treinamento inicial:

- Training accuracy ≈ 99%
- Test accuracy ≈ 65%

Esse comportamento indica **overfitting**, onde o modelo memoriza os dados de treino e apresenta baixa capacidade de generalização.

## Data Augmentation

Para mitigar o overfitting, foi adicionada uma camada de **data augmentation** ao modelo utilizando:

- `RandomFlip('horizontal')`
- `RandomRotation(0.1)`
- `RandomZoom(0.1)`

Essas transformações geram variações das imagens durante o treinamento, aumentando a diversidade dos dados sem necessidade de coletar novas imagens.

## Treinamento

O modelo foi treinado por **12 epochs** utilizando os dados escalados (`X_train_scaled`).

A camada de data augmentation é aplicada apenas durante o treinamento.

## Objetivo

Melhorar a capacidade de generalização do modelo e aumentar a accuracy no conjunto de teste através da criação dinâmica de variações das imagens de treino.
