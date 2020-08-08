---
layout: post
title: Action Branching Architectures for Deep Reinforcement Learning
category_num: 10
keyword: '[Branching DQN]'
---

# 논문 제목 : Action Branching Architectures for Deep Reinforcement Learning

- Arash Tavakoli, Fabio Pardo, Petar Kormushev
- 2019
- <https://arxiv.org/abs/1711.08946>
- 2019.10.15 정리

## Summary

- 여러 개의 action을 동시에 결정해야하는 문제에서는 기존의 discrete action algorithm으로 해결하는 데에 어려움이 있다.
- Branching Dueling Q-learning(BDQ) 이러한 문제를 해결하기 위해 결정해야 하는 action 만큼 neural network를 만드는 방법이다.
- 개별 branch에서 최적의 action을 결정하고, TD-error를 계산한다. 그리고 이 TD-error를 모두 합해 전체 loss를 구하게 된다.

## 내용 정리

### high-dimensional action task가 가지는 문제점

- 여기서 말하는 high-dimensional action task란 선택해야 하는 action의 개수가 여러 개인 상황을 의미한다.
- DQN 등과 같은 Discrete-action algorithm으로는 이와 같은 문제를 해결하는데에 어려움이 많다. 왜냐하면 가능한 action 조합의 수가 많아지면 많아질수록 효과적인 탐험이 어려워지기 때문이다.
  - "such a large action spaces are difficult to explore efficiently"(Lillicrap, 2016)
  - 차원의 저주 문제와 비슷하다고 생각하면 된다.
- 본 논문에서는 이러한 문제를 해결하기 위해 Branching Deuling Q-network(BDQ)을 제시한다.

### individual network branches

- BDQ의 가장 핵심적인 개념으로, branch란 각각 개별 action을 선택하는 neural network를 말한다.
- 즉 action dimension의 수만큼 branch가 존재하며 각각의 branch는 value를 극대화할 수 있는 action을 선택하게 된다.
  - "The core notion of the rpoposed architecture is to distribute the represenatation of th action controller across indicidual network branches"
- 자연에서도 비슷한 예를 찾을 수 있다고 하며, 논문에서는 문어를 예시로 들고 있다.

### branching deuling Q-network

- dueling DQN에 branch의 개념을 적용한 것이다.
- 즉 dueling DQN 처럼 state value 와 advantage 개념을 이용하며, state value를 결정하는 네트워크와 advantage를 결정하는 네트워크가 분리되어 있다. 이때 advantage는 각 action dimension에 있어 개별적으로 각각의 branch를 통해 구해진다.
- 이때 TD-error를 구하는 방법이 문제가 된다. dueling DQN에서는 state value를 구하는 네트워크 하나, advantage를 구하는 네트워크 하나, 두 개로 이루어져 있었지만 여기서는 advantage를 구하는 네트워크가 복수로 존재하기 때문이다.

#### 개별 branch에서 q-value를 구하는 방법

- Dueling DQN 논문에서는 각 branch에서 q-value를 구하기 위한 방법들을 제시하고 있다.
- 본 논문에서는 여러 방법 중 aggregation method가 가장 좋은 성능을 보였다고 한다.
  - aggregation method외에 local maxumum reduction method, maive alternative 등이 있다.

#### 전체 loss를 구하는 방법

- 전체 loss를 구하기 위해서는 개별 branch의 TD-error를 합해야 한다.
- 가장 단순한 방법은 각 branch의 TD-error를 모두 더해서 branch 갯수만큼 나누는 방법, 즉 평균을 이용하는 방법이다.
- 하지만 이 방법은 음수와 양수를 구별하지 않고 모두 더하기 때문에 loss의 크기를 실제보다 줄어들게 한다.
- 따라서 TD-error의 절대값을 더하여 평균내는 방법을 사용하는 것이 좋다.
- 논문에서는 절대값이 아닌 MSE를 사용하는 것이 성능이 더 좋다고 한다.
  - "the mean squared TD-error across the banches mildly enhances the performance"
