---
layout: post
title: Striving for Simplicity in Off-Policy Deep Reinforcement Learning
category_num : 2
---

# 논문 제목 : Striving for Simplicity in Off-Policy Deep Reinforcement Learning

- Rishabh Agarwal 등
- 2019
- [paper link](<https://arxiv.org/abs/1907.04543>) [git link](<https://github.com/google-research/batch_rl>)
- 2019.12.23 정리

## 세 줄 요약

- 환경과의 실시간 상호작용 없이 데이터셋을 통해 agent를 학습시키는 방법을 batch RL 또는 offline learning이라고 한다.
- offline learning을 이용해 기존의 online learning 보다 더 높은 성능의 알고리즘을 만들 수 있다. DQN으로 해결하는 discrete 환경 뿐만 아니라 DDPG를 사용하는 continous 환경에서도 검증했다.
- distributional RL 알고리즘과의 비교를 통해 일반적인 DQN 알고리즘이 exploitation 성능이 부족하다는 것을 확인하고, 나아가 알고리즘적 개선을 통해 이 부분을 해결할 수 있다는 점을 강조한다.

## 내용 정리

### Batch RL(Offline RL)

off-policy 알고리즘은 on-policy와 달리 실제 real world log data를 통해 학습이 가능하다는 장점이 있다. 이때 simulator와 상호작용하며 즉각적으로 state, action, reward 등을 주고받는 일반적인 강화학습과 달리(online) 기존에 만들어진 데이터를 이용하여 policy를 학습시키는 방법을 offline learning 또는 Batch Reinforcement Learning이라고 한다. 실제 환경이 아닌 기존에 만들어진 데이터를 이용하여 학습이 가능하다는 점에서 offline learning은 online learning에 비해 안전하다는 장점이 있다.

offline learning은 환경과의 실시간 상호작용이 아닌 주어진 데이터 셋을 바탕으로 학습을 진행한다. 따라서 replay buffer와 exploration과 관련된 부분이 필요없다. 이러한 점 때문에 online learning에 비해 실험이 단순하고 재현(reproduce)도 쉽다고 할 수 있다.

사실 replay buffer를 사용하는 off-policy algorithms 은 어느 정도 offline learning의 특성을 가진다고 할 수 있다. 유한의 replay buffer를 사용한다는 것은 전체 transaction data가 있다는 것을 가정하고 window의 위치만 바꿔가며 random sampling 하는 것과 동일하기 때문이다.

offline learning의 대표적인 문제는 다른 policy에 따라 수집된 transaction으로 학습이 이뤄지면 on-policy experience가 어려워져 Q-learning 알고리즘의 성능이 떨어진다는 것(Zhang and Sutton, 2017)이다. 이와 같은 문제는 online policy에서도 replay buffer가 매우 크면 발생할 수 있다. 즉 replay buffer의 첫 부분에 있는 transaction의 경우 학습이 진행되기 전 Policy에 의해 수집된 것인 만큼 현재의 policy에 따라 얻어진 것과는 다른 특성을 가지기 때문이다.

### Distributional RL

이 내용은 다른 논문으로 정리한 이후 추가하기

### Performance of offline learning

#### 1. Offline learning on discrete control

논문의 주요 목표 중 하나는 offline 로 off-policy 를 성공적으로 학습시키는 것이 가능한지 확인해보는 것이라고 한다. 구체적으로 논문에서는 offline learning으로 학습된 DQN 모델로 기본적인 online learning DQN 모델만큼의 성능이 나오는지 확인하고 있으며, 나아가 최근에 새롭게 등장한 off-policy DQN 계 모델로 더 나은 exploit가 가능한지에 대한 실험도 진행했다. 여기서 말하는 새로운 DQN 모델이란 distributional reinforcement learning algorithm중 하나인 distributional QR-DQN 이다.

각각의 atari game에 대해 seed를 바꿔가며 5번씩 실험을 했고, 5천만 step을 진행하며 얻은 모든 (state, action, reward, next state) tuple 을 저장해 데이터셋을 구성했다. 그리고 이를 이용하여 DQN과 QR-DQN을 이용한 offline learning을 학습시켜 online DQN과 비교하는 실험을 진행했다고 한다.

실험의 결과부터 이야기하면, `offline QR-DQN > offline DQN > online DQN` 순으로 성능이 좋았다고 한다. 논문에서는 이러한 결과를 두고 두 가지 결론을 내리고 있다. 하나는 DQN의 off-policy data exploiting이 비효율적이라는 것이고, 다른 하나는 offline learning 만으로도 강력한 Atari game agent를 만드는 것이 가능하다는 것이다.

#### 2. Offline learning on continuous control

[Fujomoto 등(2019)](<https://arxiv.org/abs/1812.02900>)에 따르면 offline learning으로 off-policy algorithm을 학습시키는 것은 비효율적이라고 한다. Fujimoto의 논문은 DDPG와 같은 기본적인 모델로만 실험을 진행했고, TD3, SAC와 같은 최신의 모델들은 다루지 않았다.

Continuous 환경에서도 offline learning의 성능을 실험하기 위해 논문에서는 DDPG를 이용해 백만 개의 transition을 모두 저장해 데이터셋을 구성했다고 한다. 그리고 이 데이터셋을 이용해 TD3과 DDPG를 offline learing 으로 학습했고, 그 결과 offline TD3이 offline DDPG 뿐만 아니라 데이터 수집에 사용된 online DDPG보다 성능이 좋았다고 한다.

#### 3. Experiment Details

실험에 사용한 hyperparameter는 기본적으로 Dapamine baselines(Castro et al. 2018)을 따랐다고 한다. 그리고 Atari game의 environment 면에서는 sticky action([Machado et al., 2018](<https://arxiv.org/abs/1709.06009>)) 을 도입하여 매 step 마다 일정 확률로 새로운 action 대신 한 시점 이전에 선택한 action을 한 번 더 하도록 하여 확률성을 높혔다. 기본적인 코드는 모두 [깃](<https://github.com/google-research/batch_rl>)에서 확인할 수 있다.

DQN replay dataset을 얻기 위해 일반적인 online DQN에 RMSprop optimizer를 사용했고, 5천만 step을 진행하며 얻어지는 transaction을 모두 수집한다. 따라서 한 번의 실험을 통해 얻은 dataset은 약 5천만 개의 experience tuple로 이뤄져 있다. 그리고 이것을 순차적으로 100만 개씩 나누어 각각 다른 파일로 저장했다.

Offline learing 실험을 위해 사용된 DQN 알고리즘에서는 ADAM을 사용했다.
