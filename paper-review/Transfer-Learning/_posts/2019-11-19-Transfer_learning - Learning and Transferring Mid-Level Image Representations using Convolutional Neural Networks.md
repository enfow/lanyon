---
layout: post
title: Transfer Learning) Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks
---

# 논문 제목 : Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks

- Maxime Oquab, Leon Bottou 등
- 2014
- <https://ieeexplore.ieee.org/document/6909618>
- 2019.11.19 정리

## 세 줄 요약

- 큰 데이터 셋을 통해 학습한 CNN 모델을 이용하여 다른 이미지 인식 문제를 효율적으로 해결할 수 있다.

## 내용 정리

### data-hungry nature

- CNN을 이용하면서 이미지 인식 성능이 매우 높아졌다. 하지만 CNN의 경우 매우 많은 파라미터를 가지고 있고, 학습을 통해 업데이트 해주어야 한다. 따라서 데이터 셋의 크기가 성능에 중요한 영향을 미치게 되는데, 이미지의 경우 많은 수의 데이터를 확보하는 것에 어려움이 있다.
- 논문에서는 transfer learning을 제안하는 이유로 이와 같이 학습 데이터의 수가 부족한 경우를 들고 있다.

### Transfer learning

- Transfer learning의 핵심적인 아이디어는 방대한 데이터셋(ImageNet)으로 학습된 pre-trained model을 중간 수준의 포괄적 추출기(**generic extractor of mid-level image representation**), 즉 일반적으로 활용될 수 있는 이미지 특징 추출기로 사용하겠다는 것이다.
- 쉽게 말해 다른 데이터셋을 가지고 학습한 model을 이용하여 문제를 해결하는 것이다. 이를 두고 논문에서는 "Transfer learning aims to transfer knowledge between related source and target domains"라고 표현한다.

### Transfer learning의 방법

#### 1. Source task 학습

- source task란 실제 해결하고자 하는 문제가 아닌, pre-trained model을 얻기 위해 이뤄지는 학습을 말한다.
- source task에 사용되는 데이터셋은 수가 많아야하고, 정확한 레이블링이 되어 있어야 한다.
- 논문에서는 ImageNet 데이터셋을 분류하는 모델을 생성하는 것으로 source task를 설정했다.

#### 2. transfering pre-trained model

- source task에서 학습한 모델에서 일반적으로 마지막 레이어는 사용하지 않는다. 마지막 레이어의 경우 이미지의 특징보다는 task의 특성(classificatiom ,prediction...)에 더 많은 영향을 미치기 때문이다.
- 마지막 레이어를 대신해 **adaptation layer**를 모델의 마지막에 붙여준다. 이들 레이어는 target task 학습을 통해 학습되어진다.
- 논문에서는 pre-trained model의 8 layer 중 마지막 1개의 레이어를 제거하고, 이를 대신하여 2개의 adaptation layer를 추가했다.

#### 3. target task 학습

- 2에서 만든 모델을 가지고 target task의 데이터셋을 이용해 학습을 진행한다.
- target task를 학습하는 과정에서는 pre-trained model은 고정된다(kept fixed).
- 이때 문제는 source task의 데이터와 target task의 데이터 간에 특성 차이가 존재한다는 점이다. 아래에서 확인할 수 있듯 이러한 문제를 dataset capture bias 라고 한다.

### Transfer learning에서 해결해야 할 문제

#### label bias

- source data와 target data의 레이블에 차이로 인해 발생하는 문제를 말한다. 예를 들어, 한 데이터셋에서는 강아지 사진을 dog로만 분류하는 반면, 다른 데이터 셋에서는 강아지 사진을 huskey, austrailianterrier 등과 같이 여러 종으로 분류하는 경우가 있다.
- 이와 같은 문제를 해결하기 위해 pre-trained model에서 마지막 레이어는 사용하지 않고, 학습되지 않은 새로운 adaptation layer로 대체하는 방법을 고안했다고 한다.

#### dataset capture bias

- 이미지 자체의 차이로 인해 발생하는 문제이다. 어떤 데이터 셋은 어떤 특정 물체에만 집중하여 이미지의 대부분을 한 물체가 채우고 있지만, 다른 데이터 셋의 이미지에는 하나에 여러 물체들이 포함되어 있다면 분류 문제를 해결하는 데에 어려움이 발생한다.
- 이러한 문제를 해결하기 위해 논문에서 제안하는 방법이 **sliding window strategy** 이다.

#### sliding window strategy

- 하나의 target data 상에서 여러 크기의 window를 이동시켜가며 patch를 추출하고, patch를 단위로 하여 레이블링과 리스케일링을 통해 학습이 가능하도록 하는 방법을 말한다.
- 특정 물체가 전체 patch에서 차지하는 비중이 작거나, 부분적으로 존재하는 경우에는 background로 판단하게 되며, 동시에 두 개 이상의 물체가 일정 크기 이상으로 존재하면 해당 patch는 학습 대상에서 제외한다. 
- 논문에서는 8 종류의 window를 이용하여 하나의 target data에서 500여 개의 patch를 추출했다고 한다.

#### dealing with background

- sliding window strategy를 사용할 경우 background로 분류되는 patch가 다수 발생하게 된다는 문제가 발생한다. 이는 곧 학습 데이터 불균형 문제로 이어진다.
- 이를 해결하기 위해 학습 과정에서의 gradient를 조정한다고 한다.
  - "This can be addressed by re-weighting the training cost function, which would amount to re-weighting its gradients during training."
