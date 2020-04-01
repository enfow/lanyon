---
layout: post
title: DSVDD) Deep_One_class_Classification
---

# 논문 제목 : DSVDD) Deep_One_class_Classification

- Patrick Schlachter 등
- 2019
- [paper link](<https://arxiv.org/abs/1902.01194>)
- 2019.11.19 정리

## Summary

- 기존에 anomaly detection 에 사용되던 SVDD 기법에 Deep neural network를 적용하여 Deep SVDD를 제안한다.
- DSVDD는 normal data를 작은 크기의 hypereshpere 내에 매핑하도록 네트워크를 학습하고, 이 과정에서 찾아낸 hypersphere를 기준으로 anomalous data를 찾아낸다.
- AutoEncoder의 reconstruction error를 이용하여 anomaly detection 하는 방법들이 주로 사용됐는데, 이러한 방법들은 직접적으로 anomaly detection을 해결하는 방법이 아니므로 DSVDD가 더 좋다고 주장한다.

## Anomaly detection 문제

- Anomaly detection 문제를 해결하는 방법으로 unsupervised learning, 즉 normal data만으로 학습을 완료하고 테스트 과정에서 한 번도 본 적이 없는 데이터가 들어왔을 때 이를 감지해내는 방법이 자주 사용된다. 이를 위해서는 정확하게 정상 데이터의 분포를 묘사하는 것이 중요하다(describes normality).
- 이러한 방법을 **One Class classification** 이라고 한다.
- 전통적인 Anomaly detection 문제를 해결하는 방법으로는 OC-SVM(One Class Support Vector Machine), KDE(Kernel Density Estimation) 등이 있지만, 데이터의 차원 수가 많으면 쉽게 차원의 저주 문제에 빠지게 되어 좋은 성능을 보여주지 못했다.
- 최근에는 Deep learning을 이용한 기법들이 Anomaly detection 문제에 좋은 성능을 보여주고 있다(promising results).

## Kernel-based One-Class Classification

### OC-SVM

- kernel-based one-class classification의 대표적인 예시가 OC-SVM이다.
- OC-SVM은 feature space에서 maximum margin hyperplane을 찾는 것을 목표로 한다.

### SVDD

- SVDD란 Support Vector Data Description 의 줄임말이다.
- OC-SVM과 가장 큰 차이는 구분하기 위한 기준으로 hyperspace가 아닌 hypershpere를 사용한다는 점이다.
  - sphere이기 때문에 center c 와 radius r 개념이 사용된다.

## Deep SVDD

- Deep SVDD(DSVDD)란 Deep Support Vector Data Description 를 뜻하며, deep neural network를 이용한 kernel-based one-class classification 방법이다.
- DSVDD는 training 과정에서 들어오는 data들의 network representation들을 최소 크기의 hypersphere 내에 밀집시키도록 network를 학습한다.
  - "Deep SVDD learns to extract the common factors of variation of the data distribution by training a neural network to fit the network outputs into a hypershpere of minimum vloume."
- 이를 위해 DSVDD는 normal data의 representation들이 특정한 hypershpere center에 최대한 가깝게 위치하도록 학습시키는데, 이 과정에서 네트워크가 데이터들의 공통점을 학습할 수 있다고 한다
  - "forces the network to extract the common factor of variation since the network must closely map the data points to the center of the sphere."
- 딥러닝을 이용하는 다른 Anomaly detection 들은 reconstruction error에 의존한다는 점에서 차이가 있다.

### AutoEncoder for anomaly detection

- 오토인코더는 기본적으로 차원 감소(dimensionality reduction)에 초점을 맞추고 있지, anomaly detection을 직접 목표로 삼지는 않는다.
- 오토인코더를 anomaly detection에 사용하기 위해 가장 중요한 것은 적절한 수준으로 정보를 압축하는 것이다. 하지만 unsupervised learning과 같은 환경에서는 이 적절한 수준을 찾는 것이 쉽지 않다.
- Deep SVDD에서도 오토인코더를 사용하는데, 정보가 압축되는 수준을 hypersphere의 크기를 최소화하는 과정에 포함시키므로 anomaly detection 문제에 직접적으로 연관시켰다. (?)

### Soft-boundary DSVDD

- Soft-boundary DSVDD는 뒤에서 나올 One-Class DSVDD와 달리 학습할 때 normal data와 anomalous data를 대상으로 학습을 진행한다. 그리고 목적함수에 anomalous data의 비율에 대한 정보(v-parameter)를 넣어준다는 점 또한 다른 점이다.
- soft-boundary DSVDD는 hypersphere의 크기를 최소화하는 과정에서 몇몇 representation은 경계 밖에 있는 것을 허용한다. 이때 목적함수에 전달되는 anomalous data 비율만큼 boundary 밖에 매핑되도록 하여 anomalous data만 boundary 밖에 위치할 수 있도록 한다.
  - "Hyperparameter v controls the trade-off between teh volumne of the sphere and violations of the boundary. i.e. allowing some points to be mapped outside the sphere."
- 네트워크는 normal data의 representation을 center c에 최대한 가깝게 매핑하도록 학습된다. 이 과정에서 네트워크는 normal data 간의 공통된 특성을 알게 된다.
- 결과적으로 normal data의 경우 center c에 가깝게, anomalous data는 멀게 매핑하도록 하여 anomaly detection 이 이뤄진다.

### One-Class DSVDD

- One-Class DSVDD는 training 단계에서는 normal data 만을 이용하며, data representation이 매핑되는 영역의 크기, 즉 center c를 중심으로 하는 hypersphere를 최소화하는 방향으로 학습이 이뤄진다.
- One-Class DSVDD의 loss function은 각 data representation과 center c 간의 거리가 크면 클수록 loss가 커지는 구조로 되어 있다. 앞서와 마찬가지로 center c에 가깝게 normal data representation을 매핑하는 과정에서 normal data의 특성을 네트워크가 학습하게 된다.

## Additional Study

- kernal based classification
  - Gaussian kernal
- SVM
- [OC-SVM_논문](<http://users.cecs.anu.edu.au/~williams/papers/P132.pdf>)
- SVDD
