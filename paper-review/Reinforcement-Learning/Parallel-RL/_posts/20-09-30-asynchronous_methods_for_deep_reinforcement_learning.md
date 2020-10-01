---
layout: post
title: Asynchronous Methods for Deep Reinforcement Learning
category_num: 1
keyword: '[A3C]'
---

# 논문 제목 : Asynchronous Methods for Deep Reinforcement Learning

- Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza 등
- 2016
- [논문 링크](<https://arxiv.org/abs/1602.01783>)
- 2020.09.30 정리

## Summary

- 복수의 Agent를 두고 Policy를 비동기적으로 업데이트하면 기존의 Replay Memory를 사용하는 방법들보다 더 빠르고 효율적인 학습이 가능하다.
- 동일한 환경에 대해 Parallel하게 Transaction을 수집하므로 Online Learning의 두 가지 문제, 즉 학습 데이터가 Non-Stationary하고 Correlated 되어 있다는 문제를 완화한다.
- 제시하는 Asynchronous RL Framework의 특징으로는 **Accumulating Gradient**, **Global Shared Parameter**, **Forward View** 등이 있으며, 대표적인 알고리즘으로는 **Asynchronous Advantaget Actor-Critic(A3C)**이 있다.

## Online Learning is Hard

Online Learning, 즉 Agent가 Environment와 직접적으로 상호작용하며 Policy를 업데이트하는 것은 본질적으로 불안정하며 학습이 잘 되지 않는다. 이는 Online Learning 과정에서 Environment로 부터 받아 학습의 대상이 되는 Sequence Data가 다음과 같은 두 가지 특징을 가지기 때문이다.

- Non-Stationary
- Correlated

여기서 **Non-Stationary**란 Time Series Data에서 평균, 분산 등 파라미터가 시간이 지남에 따라 계속 바뀌는 특성을 말하고, **Correlated**는 Sequence의 각 Data Point 간에 상관 관계가 존재한다는 것을 말한다. 이러한 문제를 해결하기 위해서는 일종의 안정화가 필요한데, 대표적인 것이 Replay Memory를 사용하는 것이다.

### Replay Memory

Replay Memory를 사용하면 많은 수의 Transaction을 모아두고 랜덤 샘플링하여 업데이트에 사용하기 때문에 Non-Stationary한 특성이 줄어들고, Correlation 또한 제거할 수 있다. 실제로 DQN을 비롯해 많은 강화학습 알고리즘이 Replay Memory를 사용하여 큰 성공을 거두었다. 하지만 Replay Memory를 사용하는 방법도 단점을 가지고 있는데, 첫 번째는 메모리를 많이 사용하고 업데이트에 요구되는 연산량이 많다는 것이고, 두 번째는 Q-learning과 같은 Off-Policy Learning만 가능하다는 것이다.

### Parallel Update

논문에서는 **Parallel Update**를 통해 Online Learning이 가지는 두 가지 문제, Non-Stationary와 Correlated 문제를 완화하면서도 Replay Memory의 단점을 보완할 수 있다고 언급한다. 여기서 말하는 Parallel Update란 말 그대로 복수의 Agent와 각각에 대응하는 Environment를 만들고 비동기적으로 각 Agent가 Transaction을 수집하도록 하여 이를 학습에 사용하는 방식이라고 말할 수 있다.

구체적으로 Parallel Update에서 Agent들은 파라미터를 서로 공유하기 때문에 각 Agent가 실시간으로 파라미터를 업데이트 한다면 여러 환경에서 수집되는 복수의 Transaction이 임의의 순서대로 학습에 사용되는 것과 마찬가지 효과를 얻게 된다. 따라서 상대적으로 학습 데이터들이 Stationary하면서도 데이터 포인트 간 Correlation을 줄이는 것이 가능해진다. 또한 Replay Memory가 전혀 필요하지 않기 때문에 Off-Policy는 물론 On-Policy도 가능하여 알고리즘을 선택하는 데 있어서도 상대적으로 자유롭다. 마지막으로 학습이 효율적이기 때문에 GPU를 사용하지 않고, Multi-Core CPU만으로도 기존의 방법론보다 더욱 빠르게 학습이 가능하다고 한다.

## Asynchronous RL Framework

기본적인 Parallel Update 알고리즘으로 논문에서는 다음 네 가지를 제시한다. 알고리즘을 구현하는 데 있어 신뢰 가능하면서도 자원을 적게 사용하는 것에 초점을 맞추었다고 한다.

- One-Step Q-learning
- One-Step SARSA
- n-Step Q-learning
- Advantage Actor-Critic(A3C)

**Asynchronous RL Framework**의 가장 큰 특징은 **Asynchronous Actor-Learners**이다. 비동기적인 Actor가 여러 개 존재하여 각자 학습을 진행한다는 것으로, Actor는 동일한 환경의 서로 다른 부분을 병렬적으로 탐색하게 된다. 구체적으로 다음 세 가지 특징들을 가지고 있다(Forward View는 n-step Q-learning과 A3C만 해당).

- **Accumulating Gradient**
- **Global Shared Parameter**
- **Forward View**

비동기적으로 Global Parameter를 업데이트하므로 Asynchronous RL Framework에서는 Optimizer 또한 어떻게 할지 문제 된다. 쉽게 말해 Agent마다 Optimizer를 둘 것인지, 공유할 것인지 결정해야 하는데 실험 결과 Optimizer를 공유하는 RMSprop가 가장 robust한 policy를 만들어냈다고 한다.

### One-Step Q-learning

<img src="{{site.image_url}}/paper-review/a3c_one_step_q_learning.png" style="width:25em; margin: 0px auto;" align="left">

각각의 경우 어떻게 학습이 이뤄지는 지에 대해서는 알고리즘을 보면서 확인하면 보다 쉽게 이해할 수 있다. 참고로 모든 알고리즘은 단일 Thread에서 동작하는 프로세스를 기준으로 한다.

왼쪽의 알고리즘을 보게 되면 첫 번째 줄에서 Current Value Net $$\theta$$와 Target Value Net $$\theta^-$$를 Thread 간에 공유한다는 것을 알 수 있다. 초기화 이후에 가장 먼저 수행하는 것은 탐색이며, 방법은 DQN과 크게 다르지 않다. 참고로 각각의 Actor가 서로 다른 탐색 알고리즘을 가지도록 하여 탐색의 다양성을 높이면 보다 Robust한 Policy를 만들 수 있다고 한다.

그런데 Gradient를 계산하고 Current Value Net을 업데이트하는 과정에 DQN과 차이가 있다. 알고리즘을 보면 DQN처럼 매 Step 업데이트를 하는 것이 아니라 Step마다 구해지는 Gradient를 누적해가며 조건이 충족될 때에만 업데이트를 수행한다. 이와 같이 업데이트하는 것은 Mini-Batch Update와 유사한데, 이를 통해 Agent가 다른 Agent가 수행한 업데이트를 Overwrite 할 가능성을 낮추고 연산의 효율성을 높일 수 있다고 한다.

구체적으로 업데이트는 Current Value Net을 업데이트하는 것과, Target Value Net을 업데이트 하는 것, 두 가지가 있는데 Current Value Net은 Thread의 Step Counter $$t$$를 기준으로 $$I_{AsyncUpdate}$$ 주기마다 업데이트 하고, Target Value Net은 Global Step Counter $$T$$를 기준으로 $$I_{target}$$ 주기마다 업데이트 한다.

### One-Step SARSA

One-Step SARSA는 One-Step Q-learning과 거의 유사하며, $$Q(s',a';\theta^-)$$에서 $$a'$$가 실제 Maximizing Action이 아니라 Policy에 의해 선택되었던 Action이라는 점에서만 차이가 있다.

### n-Step Q-learning

n-Step Q-learning과 One-Step Q-learning의 가장 큰 차이점은 **Forward View**이다. Forward View란 위의 알고리즘에서 내부 루프를 말하는 것으로, $$t_{max}$$ step만큼 탐색 알고리즘에 따라 탐색을 진행하고 그에 따라 얻은 $$t_{max}$$개의 Reward들을 학습에 사용하는 방법이라고 할 수 있다.

그리고 n-Step Q-learning에는 Thread 마다 고유의 파라미터 $$\theta'$$를 가진다는 점에서도 차이가 있다. 한 번 루프를 돌 때마다 가장 먼저 $$\theta'$$를 Current Value Net $$\theta$$과 동기화해주는데, 이는 Forward View를 수헹하는 과정에서 Current Value Net이 다른 Agent에 의해 변화하는 것을 방지하기 위함으로 보인다. 따라서 $$\theta$$에 대한 Gradient $$d\theta$$ 또한 $$\theta'$$를 기준으로 구하고 있다.

<img src="{{site.image_url}}/paper-review/a3c_n_step_q_learning.png" style="width:45em; margin: 0px auto;">

### Advantage Actor-Critic(A3C)

Advantage Actor-Critic은 논문에서 제시하는 방법 중 가장 성능이 좋으며, **Asynchronous Advantage Actor-Critic**, 줄여서 **A3C**라고 흔히 부르는 알고리즘이기도 하다. n-step Q-learning과 마찬가지로 Forward View를 적용하고 있고 Thread 마다 고유 파라미터도 가지고 있다. Actor와 Critic의 Gradient를 구하는 방법은 본래의 그것과 크게 다르지 않다.

논문에서는 Exploration 방법론으로 아래 식과 같이 Objective Function에 $$\pi$$의 Entropy를 더하는 방법을 제시하고 있다.

$$
\nabla_{\theta'} \log \pi(a_t \lvert s_t; \theta') (R_t - V(s_t; \theta_v) ) + \beta \nabla_{\theta'} H(\pi(s_t ; \theta'))
$$

위의 식과 같이 Entropy Regularization Term을 더해주는 방식으로 구현이 가능하다.

<img src="{{site.image_url}}/paper-review/a3c_actor_critic.png" style="width:45em; margin: 0px auto;">
