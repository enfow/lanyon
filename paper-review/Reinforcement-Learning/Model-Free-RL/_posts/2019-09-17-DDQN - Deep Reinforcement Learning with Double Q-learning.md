---
layout: post
title: Deep Reinforcement Learning with Double Q-learning
category_num : 2
keyword: '[DDQN]'
---

# 논문 제목 : Deep Reinforcement Learning with Double Q-learning

- David Silver 등
- 2015
- [논문 링크](<https://arxiv.org/abs/1509.06461>)

## Summary

- DQN의 경우 action의 value를 overestimation하는 문제가 빈번히 발생하는데, 이는 policy의 질을 떨어뜨리는 원인이 된다.
- DQN에서 action을 선택할 때 사용하는 Q function과 action을 평가할 때 사용하는 Q function을 분리하는 방법으로 이러한 문제를 줄일 수 있는데, 이를 Double Q-learning(2010)이라고 한다.
- 논문에서 제시하는 **DDQN(Double DQN)**은 네트워크를 하나만 학습하는 DQN의 구조적 특성은 유지하면서 동시에 Double Q-learning의 아이디어를 실현하는 알고리즘이다.

## over-estimation problem

DQN을 비롯한 Q-learning 알고리즘이 가지는 대표적인 문제 중 하나는 과대평가된 값을 선호한다는 것이다. 이를 **overestimation problem**이라고 한다. overestimation problem은 새로운 문제가 아니며 오히려 오랫동안 그 원인에 대한 연구가 이뤄졌었다.

본 논문의 공동저자인 Hasselt 등은 overestimation이 발생하는 원인으로 environmental noise를 들었다. 그리고 이러한 문제를 해결하기 위해 Double Q-learning Algorithm을 지난 2010년 제안했었다.

알고리즘적으로 논문에서는 Double Q-learning을 DQN에 적용하여 overestimation을 줄여주는 새로운 Q learning 알고리즘인 Double DQN, 줄여서 **DDQN**을 제시한다. 이론적으로는 environmental bias 외에 다른 여러 estimation error 또한 over estimation의 원인이 된다는 것을 규명하고 있다.

overestimation error 문제는 DQN 계열의 순수 Q-learning 알고리즘 뿐만 아니라 Actor-Critic 계열의 알고리즘의 발전 과정에서도 꾸준이 등장하는 문제이다. overestimation error와 관련해서는 [TD3 논문 리뷰](<https://enfow.github.io/paper-review/reinforcement-learning/actor-critic/2019/11/07/TD3-Addressing-Function-Approximation-Error-in-Actor-Critic-Methods/>)에서도 자세하게 다루고 있다.

## Double Q-learning

Double Q-learning의 관점에서 overestimation이 발생하는 원인은 action의 value를 측정하고 선택할 때 max operator를 사용하는 데에 있다. 이러한 문제를 해결하기 위해서는 action selection과 action evaluation을 분리(decoupling)할 필요가 있다는 것이 Double Q-learning의 기본 아이디어다.

- Q-learning

$$Y_t^Q = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t)$$

- double Q-learning

$$Y_t^{DoubleQ} = R_{t+1} + \gamma Q(S_{t+1}, argmax_aQ(S_{t+1}, a;\theta_t); \theta_t')$$

위의 target value를 구하는 수식을 통해 Q-learning과 double Q-learning을 비교해보면 보다 확실한데, 우선 둘 간에 가장 쉽게 확인할 수 있는 차이는 weight $$\theta$$의 개수다. 기본적인 Q learning에서는 1개만 사용하지만 Double이라는 표현에 맞게 Double Q-learning에서는 $$\theta, \theta'$$ 두 개를 가지고 있음을 알 수 있다.

$$\max$$와 $$argmax$$의 표현의 의미와 함께 두 수식의 차이를 살펴보면, Q-learning에서는 next state $$S_{t+1}$$ 과 가능한 모든 action $$a$$ 중 Q value가 가장 큰 값을 선택하게 된다. 하나의 과정처럼 보이지만 결국 각 action에 대한 평가와 동시에 가장 높은 action을 선택하므로 한 번에 두 가지가 모두 이뤄지는 것을 알 수 있다. 반면 Double Q-learning은 우선 $$\theta'$$를 기준으로 가장 Q value가 높을 것으로 예상되는 aciton을 선택하고 이후 $$\theta$$를 이용해 next state $$S_{t+1}$$와 선택된 action의 Q value를 구하고 있다. 즉, 선택에 사용되는 네트워크와 평가에 사용되는 네트워크를 나누고 있는 것이다.

그렇다면 Double Q-learning은 업데이트를 어떻게 할 지가 궁금해진다. 여러가지 방법이 가능할 것 같은데, 논문에 따르면 $$\theta, \theta'$$ 의 역할을 바꾸어가며 symmetric하게 업데이트하는 방법을 이용한다고 한다.

## Double DQN

Double Q-learning을 구현하려면 이론적으로는 action을 선택할 때 사용할 네트워크와 평가할 때 사용하는 네트워크 두 개가 필요하다. 하지만 DQN의 구조적 특성을 이용해 본래의 DQN 구조를 최대한 지키면서 double Q-learning의 장점을 살릴 수 있는 알고리즘을 제안하고자 한다. 이를 **Double DQN**이라고 하는데 업데이트 방식은 기본적으로 동일하지만 target value를 구하는데 있어 다음과 같이 기존 DQN과는 차이가 있다.

$$Y_t^{DQN} = R_{t+1}+\gamma \max_a Q(S_{t+1}, a; \theta_t^-)$$

$$Y_t^{DDQN} = R_{t+1}+\gamma Q (S_{t+1}, argmax_a Q(S_{t+1}, a; \theta_t), \theta_t^-)$$

DDQN의 target value식에서 가장 눈에 띄는 것이 있다면 **$$\theta_t^-$$**일 것이다. 여기서 $$\theta_t^-$$는 fixed network라는 것으로, 일정 step동안 업데이트 되지 않고 유지되는 네트워크라고 할 수 있다. 쉽게 말해 $$\theta_t$$는 (이 또한 학습 셋팅에 따라 달라질 수 있지만) 매 step 업데이트 되는, DQN과 동일한 네트워크이고 $$\theta_t^-$$는 정해진 $$t$$ step에 한 번씩 $$\theta_t$$와 같아지는 방식으로 업데이트되는 네트워크이다.

논문에 제시된 방법 외에도 DDQN을 구현하는 방법은 다양하다.

## Experiment

### experiment environmemt

- Mnih(2015)이 사용한 실험 환경과 network 구조를 거의 그대로 사용했다.
- 3개의 convolution layers와 1개의 FC를 사용한다.
- 6개의 Atari game에 대해서 DQN, DDQN 각각의 실험을 진행했다.

### experiment results

성능적으로 Double DQN이 DQN보다 value의 정확도와 policy의 질적인 면 모두에서 우수했다고 한다. 6번에 걸쳐 시드를 달리하며 실험해본 결과 학습 또한 Double DQN에서 보다 안정적으로 이뤄졌다고 한다.

## Result

1. 문제의 크기가 큰 경우(large scale problem)에는 Q-learning에서 overestimation problem이 발생할 가능성이 높다.
2. 과거에 알려진 것보다 overestimation problem은 생각보다 크고 광범위하게 발생한다.
3. Double DQN으로 이러한 문제를 극복할 수 있으며 이를 통해 보다 안정적이고 신뢰할 수 있는 학습이 이뤄질 수 있다(stable, reliable learning).
4. DQN algorithm을 조금만 바꾸고도 DDQN을 구현하는 방법을 논문에서 제시하고 있다.
5. 결론적으로 DDQN이 DQN보다 나은 policy를 찾는다.
