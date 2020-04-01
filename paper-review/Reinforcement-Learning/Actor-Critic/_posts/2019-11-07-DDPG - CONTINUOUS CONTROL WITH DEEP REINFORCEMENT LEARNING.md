---
layout: post
title: DDPG) continuous control with deep reinforcement learning
category_num: 1
---

# 논문 제목 : DDPG) continuous control with deep reinforcement learning

- P. Lillicrap 등
- 2016
- [논문 링크](<https://arxiv.org/abs/1509.02971>)
- 2019.11.07 정리

## Summary

- Q-learning model로는 continuous action space에서는 학습이 잘 되지 않는데, 이러한 문제를 해결하기 위해 deterministic policy gradient(DPG)를 기초로 하는 DDPG 모델을 제시한다.
- DDPG는 Actor-Critic을 기본 구조로 하여, DQN의 replay buffer, TD error 등을 사용한다. 그리고 DPG의 deterministic policy를 이용하여 action을 결정한다.
- deterministic policy에서 exploration을 하기 위해 action에 NOISE를 추가하고 있으며, 안정적인 학습을 위해 target network와 current network를 분리하고, 가중평균을 이용한 soft update로 target network를 업데이트한다.

## DQN(Q-learning)의 문제점

DQN 알고리즘을 통해 고차원의 observation space를 가지는 문제들은 해결됐지만 DQN은 action space가 이산적(discrete)하고 저차원(low dimension)인 경우에만 적용할 수 있다는 한계를 가지고 있었다. 각 action의 값이 연속적(continuous)인 경우에 소수의 이산적인 action value만 선택할 수 있다면 탐색에 어려움을 겪게 된다. 반면 action의 가짓수를 연속적인 값을 다룰 수 있을 정도로 늘리게 되면 차원의 저주에 쉽게 빠진다는 문제가 있다. 따라서 action space가 high dimension, continuous 할 경우 새로운 접근법이 필요한데, 논문의 저자들은 DDPG라고 하는 model-free, off-policy, actor-critic 알고리즘을 제시하고 있다.

## DDPG(Deep Deterministic Policy Gradient)

DDPG 알고리즘에는 DQN, Actor-Critic, DPG 등 기존의 여러가지 방법들이 혼합되어 있다.

### DQN approach

DDPG는 DQN과 마찬가지로 off-policy의 특성을 가진다. 즉, 현재 상태 state s에서 action을 선택하는 방법과 다음 상태 state s' 에서 action을 선택하는 방법이 다르다. 또한 DQN에서 처음 도입한 replay buffer를 사용하고 있으며, TD 방법론을 통해 네트워크를 업데이트한다는 점 또한 유사하다.

### Actor-Critic

DDPG의 구조는 Actor-Critic의 형태를 띄고 있어, policy를 결정하는 Actor와 policy를 결정하고 gradient를 구하는 데에 필요한 Q value를 계산하는 Critic으로 구성되어 있다.

### DPG

기본적으로 DDPG는 silver 등이 2014년 제시한 Deterministic Policy Gradient 방법을 사용하여 policy를 결정한다. 이 방법은 Deterministic 이라는 단어에서 유추할 수 있듯, policy가 확률적으로 action을 선택하는 것이 아니라 결정적으로 action을 선택한다. 따라서 state s가 policy에 주어지면 특정한 action value가 반환된다. 결정적이므로 policy가 같다면 동일한 state s를 입력으로 넣으면 항상 동일한 action a가 나온다. DDPG라는 이름은 DPG에 Deep이라는 접두사를 붙인 것이다.

- "the current policy by deterministically mapping states to a specific action"

### soft update

일반적인 Q learning의 경우 target Q function과 current Q function이 동일하기 때문에 수렴이 되지 않는 문제를 가지고 있다. 이러한 문제는 q network가 학습 과정에서 크게 변화하기 때문에 나타나는데, Actor-Critic 모델 또한 Critic 부분에서 Q function을 사용하고 있기 때문에 current value를 구하기 위해 사용되는 q function과 target value를 구할 때 사용되는 q function이 동일하면 유사한 문제가 나타난다. DDPG에서는 이러한 문제를 해소하기 위해 target q function을 별도로 두고 있다.

target q function이 current q function과 크게 다르지 않으면서도 안정적인 업데이트를 위해 변화량이 큰 current q function을 그대로 반영하지 않고 기존의 target q function과 현재의 current q function을 가중평균 하는 방법을 통해 target q function을 업데이트하게 되는데, 이를 soft update라고 한다.

$$
\theta' \rightarrow \tau\theta + (1 - \tau) \theta \quad \text{with} \ \tau < 1
$$

### exploration in deterministic policy

continuous action space의 문제점 중 하나는 expploration을 실시하는 방법이다. DDPG에서는 policy를 통해 결정된 action에 NOISE를 더해주는 방법으로 exploration을 실시한다.

$$
\mu'(s_t) = \mu (s_t \lvert \theta_t^\mu) + N
$$

## Algorithm of DDPG

<img src="{{site.image_url}}/paper-review/ddpg_algorithm.png">

- 1 번째 줄에서 actor와 critic을 선언하고, 이어 두 번째 줄에서 actor, critic을 복제하여 target network 를 두고 있다.
- 8 번째 줄에서 deteriministic policy $$\mu()$$를 통해 현재 state에 대한 action을 고르게 되는데, action에 NOISE를 추가해주고 있다. 여기서 NOISE는 탐색(exploration) 기능을 수행한다.
- 12 번째 줄에서 target y를 구하게 되는데, 이때 $$Q'$$과 $$\mu'$$를 사용한다. 즉, state s에서 action을 선택하는 policy와 target을 구하는 policy가 다르다. 이러한 점에서 off-policy의 특성을 가진다.
- 13 번째 줄에서 critic Q를 업데이트한다.
- 14 번째 줄에서 policy를 업데이트한다. policy 업데이트 수식은 deterministic policy gradient를 따른다.
- 15 번째 줄에서 target network를 업데이트한다. soft update 방식, 즉 기존의 target network와 current network의 가중평균을 통해 업데이트한다.
