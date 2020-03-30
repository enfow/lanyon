---
layout: post
title: Batch Norm) How does batch normalization help optimization
---

# 논문 제목 : How does batch normalization help optimization

- Shibani Santurkar 등
- 2019
- <https://arxiv.org/abs/1805.11604>
- 2019.09.14 정리

## 세 줄 요약

- Batch Normalization이 줄어드는 이유로 ICS(internal covariate shift)의 감소가 제시되었지만 확인할 수 없었다.
- ICS의 감소보다는 landscape of loss 가 보다 stable해지고 smooth 해지기 때문으로 보인다. 이는 BatchNorm을 사용했을 때 step별 loss, gradient predictiveness, beta-smoothness의 크기가 작아지는 것으로 확인할 수 있다.
- 그런데 BatchNorm외에 기본적인 statistics Regularization strategies, 즉 L1, L2 Regularization으로도 비슷한 효과를 볼 수 있었다.

## 내용 정리

### Introduction

- "At a high level, BatchNorm is a technique that aims to improve the training of neural networks by stabilizing the distribution of layer inputs"
  - BatchNorm은 각 layer의 출력값의 분포를 안정화하는 역할을 한다.

### BatchNorm과 Internal Covariate Shift

- 지금까지 Batch Normalization의 효과에 대한 설명 중 가장 잘 받아들여져 온 것으로는 ICS(Internal Covariate Shift)라는 것이 있다. BatchNorm이 ICS를 줄여주기 때문에 효과가 있다는 것이다.
- 추가 공부 필요
- 하지만 우리가 볼 때 BatchNorm은 ICS와 무관하다.

### BatchNorm과 loss landscape

- ICS 대신 loss landscape를 보다 stable하고 smooth하게 만들어준다는 설명이 보다 정확해보인다.
- "It makes the landscape of the corresponding optimization problem significantly more smoother. This ensures that the gradients are more prodictive and thus allows for use of larger of learning rate and faster networks convergence"

### 3. Why does BatchNorm work

- "It reparametrizes the underlying optimization problem to make its landscape significantly more smooth. The first manifestation of this impact is improvement in the Lipschitzness of the loss function"
  - BatchNorm의 효과를 수학적으로는 'loss function의 Lipschitzness를 증가시킨다'라고 표현할 수 있다.
- "The key implication of BatchNorm's reparametrization is that it makes the gradient more reliable and predictive"
  - 이는 곧 gradient를 보다 예측가능하게 해준다.
- "It enables any (gradient-based) training algorithm to take longer steps without the danger of running into a sudden change of the loss landscape such as flat region or sharp local minimun"
  - 그리고 이것은 특별히 loss landscape를 변화시키지 않고도 flat region과 sharp local minimun 문제를 거의 모든 gradient algorithm에 있어 쉽게 해결할 수 있도록 도와준다.
- 정리하면 다음과 같다.
    1. BatchNorm을 사용하면 Loss landscape가 smooth해진다.
    2. 이로인해 gradient가 보다 예측가능(predictive)해진다.
    3. 그 결과로 local minimun, flat region 문제가 해결하기 쉬워진다.
    4. 이로 인해 learning rate의 선택 범위가 늘어난다 + 학습이 보다 빠르게 진행된다 와 같은 이점이 생긴다.

### Exploration of the optimization landscape

- loss landscape, gradient predictiveness, beta-smoothness 모두 step별 전체적인 크기가 줄어들고 분산 또한 줄어든다.
- 각각이 정확하게 무엇을 의미하는지 찾아볼 필요가 있다.

### Is BatchNorm the best way to smoothen the landscape

- BatchNorm의 성능을 확인하기 위해 statistics based normaliztion들을 적용 비교해보았다.
  - 구체적으로 L1, L2, Maximum Norm regularization을 적용했다.
- 기본적으로 BatchNorm과 다른 statistics based normalization들은 모두 비슷한 효과를 낳았다.
  - L1의 경우 BatchNorm보다 더 좋은 성능을 보여주기도 했다.
  - "Also all these techinques result in an improved smoothness of the landscape that is similar to the effect of BatchNorm"
