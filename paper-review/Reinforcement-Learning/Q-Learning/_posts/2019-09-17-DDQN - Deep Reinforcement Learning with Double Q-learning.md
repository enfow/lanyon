---
layout: post
title: DDQN) Deep Reinforcement Learning with Double Q-learning
---

# 논문 제목 : Deep Reinforcement Learning with Double Q-learning

- David Silver 등
- 2015
- <https://arxiv.org/abs/1509.06461>
- 2019.09.17 정리

## 세 줄 요약

- DQN을 비롯한 단순한 Q-learning algorithm의 경우 action이 가지는 value를 over optimization 하는 문제가 빈번히 발생한다.
- DQN에서 action을 선택(selection)할 때 사용하는 Q function과 action을 평가(evaluation) 할 때 사용하는 Q function을 분리(decoupling)하는 방법으로 이러한 문제를 해결할 수 있다.
- 이러한 DQN algorithm을 DDQN(Double DQN)이라고 하며, 본 논문에서는 기존의 DQN을 크게 변형하지 않으면서도 Double Q-learning의 장점(over optimization의 해결)을 취할 수 있는 algorithm을 제안하고 있다.

## 내용 정리

### DQN과 over-estimation problem

- DQN의 Q function, Q(s, a; Θ)의 Θ는 weight 값을 말하며, weigth 값은 state의 space n X action의 space m의 크기로 되어 있다.
- 이러한 DQN은 use of a target network와 use of experience replay를 특성을 한다.
- DQN을 비롯한 Q-learning 알고리즘이 가지는 대표적인 문제 중 하나는 과대평가된 값을 선호한다는 것이다. 이를 overestimation problem이라고 하는데, 이는 Q-learning에서 value를 maximize하는 action을 항상 선택하기 때문이다.
- 본 논문의 공동저자인 Hasselt 등(2010)은 이러한 overestimation problem이 environmental bias 때문에 발생한다고 보고, 이를 해결하기 위한 방법으로 Double Q-learning을 제안했다.
- 추가적으로 본 논문에서는 environmental bias 외에 다른 여러 estimation error(function approximation, non-stationarity and any other sources) 또한 over optimization의 원인이 된다는 것을 규명하려고 한다.

### Double Q-learning

- Q-learning에서 overestimation이 발생하는 이유는 action의 value를 측정(evaluation)하고 선택(selection)하는 데에 max operator를 사용하기 때문이다. 즉 overestimated 된 action 이 많이 선택될 수 밖에 없기 때문이다.
- 이러한 문제를 막기 위해 action selection과 action evaluation을 분리(decoupling)하는 것이 Double Q-learning의 기본 아이디어이다.

### target Y의 비교

- Q-learning

    `Y𝗍 ≡ R𝗍₊₁ +γmaxQ(S𝗍+1,a;θ𝗍)`

- DQN

    `Y𝗍 ≡R𝗍₊₁+γmaxQ(S ,a;θ−)`

- double Q-learning

    `Y𝗍 = R𝗍₊₁ + γQ(St+1, argmax𝘢Q(St+1, a; θt); θ't)`

- double DQN

    `Y𝗍 ≡ R𝗍₊₁ + γQ(St+1 ,argmax𝘢Q(S ,a;θt);θ⁻t)`

- Q-learning은 evaluation과 selection에 있어 모두 동일한 parameter θt 를 사용하지만 double Q-learning은 selection을 위해 argmax𝘢Q(St+1, a; θt)을 사용하고, evaluation을 위해 θ't를 사용한다는 점에서 차이가 있다.
- 이 경우 θ와 θ'을 각각의 경우에 번갈아가며 사용하도록 하여 symmetric하게 업데이트가 되도록 한다. ("we use the second set of weights θt′ to fairly evaluate the value of this policy. This second set of weights can be updated symmetrically by switching the roles of θ and θ′.")
- double DQN에서는 double Q-learning과 달리 θ⁻t 를 사용하고 있다. 여기서 θ⁻t는 SGD에 의해 업데이트 되는 parameter이다.

### Double DQN

- "The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation"
- double DQN은 이론적으로는 selection에 사용되는 parameter와 evaluation에 사용되는 parameter를 서로 달리하여 두 개의 weight를 사용해야 한다.
- 하지만 본 논문에서는 DQN의 구조적인 특성을 이용하여 DQN의 본래 구조를 최대한 지키면서 double Q-learning의 장점을 얻을 수 있는 알고리즘을 사용하고자 한다.
- 본 논문에서는 다음과 같이 두 개의 parameter를 분리하여 업데이트한다.
  - selection weight는 매 sample에 대해서 weight 값을 업데이트 한다.
  - evaluation weight는 매 batch에 대해서 weight 값을 업데이트 한다.
- 결과적으로 매 sample에 대해 하나의(selection의) weight만 업데이트하면서 batch가 끝나면 evaluation weight를 selection weight로 업데이트하는 방식으로 구현할 수 있게 된다.
- 구체적인 구현 방식은 다양하다.

### 실험 환경

- Mnih(2015)이 사용한 실험 환경과 network 구조를 거의 그대로 사용했다.
- 3개의 convolution layers와 1개의 FC를 사용한다.
- 6개의 Atari game에 대해서 DQN, DDQN 각각의 실험을 진행했다.

### 실험 결과

#### 성능

- Double DQN이 DQN보다 value accuracy, policy quality의 면에서 보다 우수했다.
  - Double DQN의 estimate와 true value of the final policy 간의 차이가 DQN의 그것보다 작았다.
  - Double DQN의 true value of the final policy가 DQN보다 컸다.
- 학습 또한 Double DQN에서 보다 안정적으로 진행됐다.
  - random seed를 달리하여 6번의 실험을 진행한 결과 estimate의 값의 폭이 Double DQN보다 DQN에서 더 컸다.

#### over optimization의 관점에서의 성능

- DQN의 경우 over optimization 문제가 발생하여 resulting policy의 질을 떨어뜨리는 것으로 보인다.
  - DQN의 value estimation과 score를 비교해 볼 때, score가 떨어지는 지점에서 value estimation이 증가하는 것을 확인할 수 있다.
  - 그리고 score가 떨어진 이후 estimation의 값의 폭이 보다 커지는 것 또한 확인할 수 있었다.

- Double DQN의 학습이 보다 안정적으로 이뤄지는 것을 확인할 수 있다.
  - DQN과 비교해 볼 때 Double DQN의 estimation 값의 폭이 작다.
  - value optimization 문제가 적게 발생하여 score가 보다 높다.

### 결론

1. 문제의 크기가 큰 경우(large scale problem)에는 Q-learning에서 over optimization problem이 발생할 가능성이 높다.
2. 과거에 알려진 것보다 over optimization problem은 생각보다 크고 광범위하게 발생한다.
3. Double DQN으로 이러한 문제를 극복할 수 있으며 이를 통해 보다 안정적이고 신뢰할 수 있는 학습이 이뤄질 수 있다(stable, reliable learning).
4. DQN algorithm을 조금만 바꾸고도 DDQN을 구현하는 방법을 논문에서 제시하고 있다.
5. DDQN이 DQN보다 나은 policy를 찾는다.
