---
layout: post
title: DDPG) CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING
---

# 논문 제목 : CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING

- P. Lillicrap 등
- 2016
- <https://arxiv.org/abs/1509.02971>
- 2019.11.07 정리

## 세 줄 요약

- Q-learning model로는 continuous action space에서는 학습이 잘 되지 않는데, 이러한 문제를 해결하기 위해 deterministic policy gradient(DPG)를 기초로 하는 DDPG 모델을 제시한다.
- DDPG는 Actor-Critic을 기본 구조로 하여, DQN의 replay buffer, TD error 등을 사용한다. 그리고 DPG의 deterministic policy를 이용하여 action을 결정한다.
- deterministic policy에서 exploration을 하기 위해 action에 NOISE를 추가하고 있으며, 안정적인 학습을 위해 target network와 current network를 분리하고, 가중평균을 이용한 soft update로 target network를 업데이트한다.

## 내용 정리

### DQN(Q-learning)의 문제점

- DQN(Q-learning)은 고차원(high dimension)의 observation space를 가지는 문제들은 해결했지만 action space 가 이산적(discrete)이고 저차원(low dimension)인 경우에만 적용이 가능하다는 단점이 있다.
- Q value를 기준으로 action을 선택하는 Q-learning 모델들의 경우 action의 종류가 continuous할 경우 소수의 discrete action 만으로는 탐색에 어려움을 겪게 된다. 이를 피하기 위해 continuous action space에 맞춰 수 많은 discrete action 을 상정하면 차원의 저주에 빠지게 된다.
- 따라서 action space가 high dimension, continuous 할 경우 새로운 접근법이 필요한데, 논문의 저자들은 DDPG라고 하는 model-free, off-policy, actor-critic 알고리즘을 제시하고 있다.

### DDPG(Deep Deterministic Policy Gradient)

- DDPG에는 DQN, Actor-Critic, DPG 등 기존의 여러가지 방법들이 혼합되어 있다.

#### DQN approach

- DDPG는 DQN과 마찬가지로 off-policy의 특성을 가진다. 즉, 현재 상태 state s에서 action을 선택하는 방법과 다음 상태 state s' 에서 action을 선택하는 방법이 다르다.
- replay buffer를 사용하여 transaction간의 상관관계를 최소화한다.
- Temporal Difference를 이용하여 network를 업데이트한다.

#### Actor-Critic

- DDPG의 구조는 Actor-Critic의 형태를 띄고 있어, policy를 결정하는 Actor와 policy를 결정하고 gradient를 구하는 데에 필요한 Q value를 계산하는 Critic으로 구성되어 있다.

#### DPG

- silver 등이 2014년 제시한 Deterministic Policy Gradient 방법을 사용하여 policy를 결정한다.
- 이 방법은 Deterministic 이라는 단어에서 유추할 수 있듯, policy를 stochastic하게 선택하는 것이 아니라 deterministic하게 결정하는 것이다.
- 따라서 state s가 주어지면 특정 action이 확률적으로 정해지는 것이 아니라 하나의 action만이 나오도록 Actor가 동작한다. 즉, 모델이 같다면 state s를 넣으면 항상 동일한 action a가 나오게 된다.
  - "the current policy by deterministically mapping states to a specific action"
- DPG에 DQN의 Deep을 붙여 DDPG라는 이름이 나오게 되었다.

#### soft update

- 일반적인 Q learning의 경우 target Q function과 current Q function이 동일해 수렴이 되지 않는 문제가 자주 발생한다고 한다(overestimation bias?).
- 이와 같은 문제는 학습과정에서 Q function이 크게 변화하고, 이 때문에 target Q value 또한 step마다 크게 변화하기 때문이다.
- 이러한 문제를 해결하기 위해 도입한 것이 soft update이다.
- soft update는 target Q function과 current Q function을 구분하며, current Q function은 기존의 방식대로 update하되, target Q function은 current Q function과 기존 target Q function 간의 가중평균을 이용해 업데이트하는 방법이다.
- 이를 이용하면 target Q function의 변화속도가 느려져, 안정적인 학습에 도움이 된다.

    `
    θ′ ←τθ +(1−τ)θ′ with τ < 1
    `

#### exploration in deterministic policy

- continuous action space의 문제점 중 하나는 expploration을 실시하는 방법이다.
- DDPG에서는 policy를 통해 결정된 action에 NOISE를 더해주는 방법으로 exploration을 실시한다.

    `
    μ′(st) = μ(st|θtμ) + N
    `

### Algorithm of DDPG

```
Randomly initialize critic network Q(s, a|θQ) and actor μ(s|θμ) with weights θQ and θμ.
Initialize target network Q′ and μ′ with weights θQ′ ← θQ, θμ′ ← θμ
Initialize replay buffer R
for episode = 1, M do
    Initialize a random process N for action exploration
    Receive initial observation state s1
    for t = 1, T do
        Select action a𝗍 = μ(s𝗍|θμ) + NOISE𝗍 according to the current policy and exploration noise
        Execute action a𝗍 and observe reward r𝗍 and observe new state s𝗍₊₁
        Store transition (s𝗍, a𝗍, r𝗍, s𝗍₊₁) in R
        Sample a random minibatch of N transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from R

        Set yᵢ
            yᵢ = rᵢ + γQ′(sᵢ₊₁, μ′(sᵢ₊₁|θμ′ )|θQ′ )

        Update critic by minimizing the loss:
            L = (1/N)*(yᵢ − Q(sᵢ, aᵢ|θQ))²

        Update the actor policy using the sampled policy gradient:
            ∇θμ J ≈ (1/N) ∇aQ(s, a|θQ)|s=sᵢ,a=μ(sᵢ)∇θμ μ(s|θμ)|sᵢ

        Update the target networks:
            θQ′ ←τθQ +(1−τ)θQ′
            θμ′ ←τθμ +(1−τ)θμ′

    end for
end for
```

- 1 번째 줄에서 actor와 critic을 선언하고, 이어 두 번째 줄에서 actor, critic을 복제하여 target network 를 두고 있다.
- 8 번째 줄에서 deteriministic policy μ()를 통해 현재 state에 대한 action을 고르게 되는데, action에 NOISE를 추가해주고 있다. 여기서 NOISE는 탐색(exploration) 기능을 수행한다.
- 12 번째 줄에서 target y를 구하게 되는데, 이때 Q'과 μ'를 사용한다. 즉, state s에서 action을 선택하는 policy와 target을 구하는 policy가 다르다. 이러한 점에서 off-policy의 특성을 가진다.
- 13 번째 줄에서 critic Q를 업데이트한다.
- 14 번째 줄에서 policy를 업데이트한다. policy 업데이트 수식은 deterministic policy gradient를 따른다.
- 15 번째 줄에서 target network를 업데이트한다. soft update 방식, 즉 기존의 target network와 current network의 가중평균을 통해 업데이트한다.
