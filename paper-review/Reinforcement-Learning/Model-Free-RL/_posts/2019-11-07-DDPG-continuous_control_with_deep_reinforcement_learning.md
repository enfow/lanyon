---
layout: post
title: Continuous Control with Deep Reinforcement Learning
category_num: 11
keyword: '[DDPG]'
---

# 논문 제목 : DDPG) Continuous Control with Deep Reinforcement Learning

- P. Lillicrap 등
- 2016
- [논문 링크](<https://arxiv.org/abs/1509.02971>)

## Summary

- Continuous Action Space에서도 높은 성능을 보이는 새로운 **Model-Free, Off-Policy, Actor-Critic** 강화학습 알고리즘 **DDPG**를 제안한다.
- DDPG는 **Deterministic Policy Gradient(DPG)**에 이론적 근거를 두고 있으며, 이를 구성하는 두 함수를 Neural Net으로 모사하게 된다.
- Neural Net을 사용하여 발생할 수 있는 문제들을 DQN에서 도입된 기법들(Replay Buffer, Target Network)을 사용하여 해결한다.

## Limitations of DQN

DQN 알고리즘은 Observation Space가 고차원인 경우에 대해서도 높은 성능을 보였지만 Action Space가 Discrete한 경우에만 가능하다는 한계를 가지고 있다. 이는 Action Space가 Continuous 한 경우에 Action의 갯수가 무한하다보니 Action Value를 기준으로 최적의 Action을 결정하는 DQN으로는 현실적으로 모든 Action의 Value를 알 수 없기 때문이다. 따라서 Action Space가 Continuous한 경우 새로운 접근법이 필요하다. 논문에서는 이와 관련하여 DPG(Deterministic Policy Gradient) 알고리즘에 DQN에서 Neural Net 학습을 위해 적용된 여러 방법론들을 적용한 **DDPG(Deep Deterministic Policy Gradient)** 알고리즘을 대안으로 제시하고 있다.

## Deep Deterministic Policy Gradient

**DDPG**는 이름 그대로 Deterministic Policy $$\mu(s)$$를 가정할 때 이를 업데이트하는 방향(Gradient)을 구하는 [DPG(Deterministic Policy Gradient) Thoerem](<http://proceedings.mlr.press/v32/silver14.pdf?CFID=6293331&CFTOKEN=eaaee2b6cc8c9889-7610350E-DCAB-7633-E69F572DC210F301>)에 기초하고 있다. Off-Policy Deterministic Policy Gradient를 가정하는 경우 Performance Gradient는 다음과 같이 구해진다.

$$
\eqalign{
\nabla_\theta J_\beta(\mu_\theta) &\approx \int_S \rho^\beta(s) [\nabla_\theta \mu_\theta(s) Q^\mu(s,a)] ds \\
&= E_{s \backsim \rho^\beta} [\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a) \lvert_{a = \mu_\theta(s)} ]
}
$$

DDPG는 위 식에서 사용되는 Q-function(Critic) $$Q^\mu(s,a)$$와 Policy(Actor) $$\mu(s)$$를 각각 Neural Network $$Q(s,a \lvert \theta^Q)$$, $$\mu(s \lvert \theta^\mu)$$로 Approximation 하게 된다. Neural Net(Non-linear Function Approximator)을 Q-learning으로 업데이트할 경우 이론적으로 수렴성이 보장되지 않지만, DQN에서 경험적으로 안정성이 어느 정도 확인되었다고 보고 여기서도 Q-learning으로 Critic을 업데이트하게 된다.

## How to Stably Update Neural Net

DDPG에서는 Neural Net의 안정적인 학습을 위해 DQN의 방법론들을 많이 차용하고 있는데, 구체적으로 **Replay Buffer**와 **Target Network**를 사용하고 있다. 나아가 Batch Normalization을 통해 Observation에서 발생하는 Covariance Shift 문제도 해결할 수 있다고 언급한다.

### Introducing Replay Buffer

Replay Buffer를 DQN에서 도입한 이유는 크게 두 가지인데, Sample Efficiency를 높이는 것과 Transition 간 Correlation을 제거하는 것이다. Transition이 Agent가 경험한 순서대로 나타난다는 점에서 DDPG 또한 동일한 문제를 가지고 있기 때문에 i.i.d Condition을 맞춰주기 위해서 Replay Buffer를 구현하고 여기서 Sampling하여 학습에 사용하도록 하고 있다. 또한 DDPG는 Off-Policy를 가정하므로 Replay Buffer를 크게 가져가도 문제가 없다.

### Introducing Target Network

Neural Network로 Q-function을 모사할 때 Temporal Difference를 적용하게 되면 Target $$y$$에도 학습의 대상이 되는 Q Network의 출력값이 사용된다.

$$
\eqalign{
L(\theta^Q) &= E[(Q(s_t, a_t \lvert \theta^Q) - y_t)^2]\\
\text{ where } &y_t = r(s_t, a_t) + \gamma Q(s_{t+1}, \mu(s_{t+1})\lvert \theta^Q)
}
$$

이 경우 Q Network가 쉽게 발산해버리는 문제가 발생한다(Neural Net에 Q-learning을 적용할 때에는 수렴성이 보장되지 않는다). DQN에서는 이러한 문제를 해결하기 위해 Target Q Network $$Q'(s,a \lvert \theta^{Q'})$$를 도입한다.

$$
y_t = r(s_t, a_t) + \gamma Q'(s_{t+1}, \mu(s_{t+1})\lvert \theta^{Q'})
$$

DDPG의 Critic에서도 동일한 문제를 가지고 있으므로 Target Q Network를 도입하고 있다. 여기서 Target $$y$$의 계산에는 Policy $$\mu$$의 영향도 받는데, $$\mu$$에 대해서도 Target Network $$\mu'$$를 만들어주는 것이 학습의 안정성을 더욱 높여준다고 한다.

$$
y_t = r(s_t, a_t) + \gamma Q'(s_{t+1}, \mu'(s_{t+1} \lvert \theta^{\mu'})\lvert \theta^{Q'})
$$

그런데 DQN과 완전히 동일한 방법, 즉 주기적으로 Current Q Network를 그대로 복사하여 Target Q Network로 만들어주는 것은 Actor-Critic 구조의 특성상 학습의 안정성을 떨어뜨릴 수 있다. $$\nabla_a Q(s,a \lvert \theta^{Q})$$ 따라 Actor의 Gradient가 결정되는데 이것의 Variance가 크면 Actor의 안정적인 학습을 방해하기 때문이다. 따라서 DDPG에서는 아래 식을 따르는 **Soft Update**를 도입하여 Target Network를 업데이트한다.

$$
\theta' \rightarrow \tau\theta + (1 - \tau) \theta \quad \text{with} \ \tau < 1
$$

### Introducing Batch Normalization

환경에 따라서는 Observation을 구성하는 개개 Feature의 Scale이 다를 수 있고, 학습이 진행되면서 각 Feature의 분포가 크게 달라지기도 한다. 이러한 상황에서는 Neural Net의 학습 안정성이 크게 떨어지기도 하는데, 이를 해결하기 위해 환경의 특성에 따라 Batch Normalizaton을 도입하기도 했다고 한다.

## Exploration in Deterministic Policy

Continuous Action Space에서 Deterministic Policy를 가정할 때에는 Exploration을 수행하는 방법이 문제가 된다. DDPG에서는 Policy를 통해 결정된 Action에 NOISE를 더해주는 방법으로 Exploration을 실시한다.

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
