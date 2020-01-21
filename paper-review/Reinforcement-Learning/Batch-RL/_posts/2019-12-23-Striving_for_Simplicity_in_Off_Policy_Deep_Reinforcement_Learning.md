---
layout: post
title: REM) Striving for Simplicity in Off-Policy Deep Reinforcement Learning
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
- distributional RL 알고리즘과의 비교를 통해 일반적인 DQN 알고리즘이 exploitation 성능이 부족하다는 것을 확인하고, 나아가 이를 효과적으로 해결하는 새로운 알고리즘으로 **REM**을 제시한다.

## 내용 정리

### Batch RL(Offline RL)

off-policy 알고리즘은 on-policy와 달리 실제 real world log data를 통해 학습이 가능하다는 장점이 있다. 이때 simulator와 상호작용하며 즉각적으로 state, action, reward 등을 주고받는 일반적인 강화학습과 달리(online) 기존에 만들어진 데이터를 이용하여 policy를 학습시키는 방법을 offline learning 또는 Batch Reinforcement Learning이라고 한다. 실제 환경이 아닌 기존에 만들어진 데이터를 이용하여 학습이 가능하다는 점에서 offline learning은 online learning에 비해 안전하다는 장점이 있다.

offline learning은 환경과의 실시간 상호작용이 아닌 주어진 데이터 셋을 바탕으로 학습을 진행한다. 따라서 replay buffer와 exploration과 관련된 부분이 필요없다. 이러한 점 때문에 online learning에 비해 실험이 단순하고 재현(reproduce)도 쉽다고 할 수 있다.

사실 replay buffer를 사용하는 off-policy algorithms 은 어느 정도 offline learning의 특성을 가진다고 할 수 있다. 유한의 replay buffer를 사용한다는 것은 전체 transaction data가 있다는 것을 가정하고 window의 위치만 바꿔가며 random sampling하는 것과 동일하기 때문이다.

offline learning의 대표적인 문제는 다른 policy에 따라 수집된 transaction으로 학습이 이뤄지면 on-policy experience가 어려워져 Q-learning 알고리즘의 성능이 떨어진다는 것(Zhang and Sutton, 2017)이다. 이와 같은 문제는 online policy에서도 replay buffer가 매우 크면 발생할 수 있다. 즉 replay buffer의 첫 부분에 있는 transaction의 경우 학습이 진행되기 전 Policy에 의해 수집된 것인 만큼 현재의 policy에 따라 얻어진 것과는 다른 특성을 가지기 때문이다.

### Distributional RL

이 내용은 다른 논문으로 정리한 이후 추가하기

### Performance of offline learning

#### 1. Offline learning on discrete control

논문의 주요 목표 중 하나는 offline 로 off-policy 를 성공적으로 학습시키는 것이 가능한지 확인해보는 것이라고 한다. 구체적으로 논문에서는 offline learning으로 학습된 DQN 모델로 기본적인 online learning DQN 모델만큼의 성능이 나오는지 확인하고 있으며, 나아가 최근에 새롭게 등장한 off-policy DQN 계 모델로 더 나은 exploit가 가능한지에 대한 실험도 진행했다. 여기서 말하는 새로운 DQN 모델이란 distributional reinforcement learning algorithm중 하나인 distributional QR-DQN 이다.

각각의 atari game에 대해 기본 DQN 알고리즘을 이용해 seed를 바꿔가며 5번씩 실험을 했고, 5천만 step을 진행하며 얻은 모든 (state, action, reward, next state) tuple 을 저장해 데이터셋을 구성했다. 그리고 이를 이용하여 DQN과 QR-DQN을 이용한 offline learning을 학습시켜 online DQN과 비교하는 실험을 진행했다고 한다.

<img src="{{site.image_url}}/paper-review/Striving_for_Simplicity_in_Off_Policy_Deep_Reinforcement_Learning_fig2.png">

위의 그림은 offline QR-DQN, offline DQN의 성능을 online DQN을 0으로 하여 로그 스케일로 비교한 것이다. 왼쪽의 그림이 offline DQN이고, 오른쪽의 그림이 offline QR-DQN인데 왼쪽은 0보다 낮은 경우가 많고, 오른쪽은 0보다 큰 경우가 많다. 이를 두고 `offline QR-DQN > online DQN > offline DQN` 순으로 성능이 좋았다고 평가한다. 논문에서는 이러한 결과를 두고 두 가지 결론을 내리고 있다. 하나는 DQN의 off-policy data exploiting이 비효율적이라는 것이고, 다른 하나는 offline 알고리즘 만으로도 강력한 Atari game agent를 만드는 것이 가능하다는 것이다.

#### 2. Offline learning on continuous control

[Fujomoto 등(2019)](<https://arxiv.org/abs/1812.02900>)에 따르면 offline learning으로 off-policy algorithm을 학습시키는 것은 비효율적이라고 한다. Fujimoto의 논문은 DDPG와 같은 기본적인 모델로만 실험을 진행했고, TD3, SAC와 같은 최신의 모델들은 다루지 않았다.

Continuous 환경에서도 offline learning의 성능을 실험하기 위해 논문에서는 DDPG를 이용해 백만 개의 transition을 모두 저장해 데이터셋을 구성했다고 한다. 그리고 이 데이터셋을 이용해 TD3와 DDPG를 offline learing 으로 학습했고, 그 결과 offline TD3가 offline DDPG 뿐만 아니라 데이터 수집에 사용된 online DDPG보다 성능이 좋았다고 한다.

#### 3. Experiment Details

실험에 사용한 hyperparameter는 기본적으로 Dapamine baselines(Castro et al. 2018)을 따랐다고 한다. 그리고 Atari game의 environment 면에서는 sticky action([Machado et al., 2018](<https://arxiv.org/abs/1709.06009>)) 을 도입하여 매 step마다 일정 확률로 새로운 action 대신 한 시점 이전에 선택한 action을 한 번 더 하도록 하여 확률성을 높혔다. 기본적인 코드는 모두 [깃](<https://github.com/google-research/batch_rl>)에서 확인할 수 있다.

DQN replay dataset을 얻기 위해 일반적인 online DQN에 RMSprop optimizer를 사용했고, 5천만 step을 진행하며 얻어지는 transaction을 모두 수집한다. 따라서 한 번의 실험을 통해 얻은 dataset은 약 5천만 개의 experience tuple로 이뤄져 있다. 그리고 이것을 순차적으로 100만 개씩 나누어 각각 다른 파일로 저장했다.

Offline learing 실험을 위해 사용된 DQN 알고리즘에서는 ADAM을 사용했다.

### REM: Random Ensemble Mixture

QR-DQN을 비롯한 distributional RL의 기법들이 batch setting에서 좋은 성능을 보여주었고, 이에 따라 알고리즘의 exploiting 성능이 뛰어나다면 dataset만 가지고도 학습이 가능하다는 결론을 내리고 있다. 하지만 distributional RL은 bellman equation을 distribution으로 표현하고, 따라서 TD error 또한 scalar 값이 아닌 분포로 표현된다는 점에서 복잡하고, 최근의 연구결과를 볼 때 distributional RL의 기여가 불분명한 측면이 있다고 한다. 따라서 논문에서는 exploiting 성능이 높은 새로운 알고리즘을 제시하고 있는데 이것이 **REM**이다.

#### Ensenble-DQN

REM은 Ensenble-DQN에서 아이디어를 얻었다고 한다. Ensenble-DQN은 앙상블이라는 표현에서 느낌이 오듯 여러 개의 agent를 동시에 학습시키고, 각각의 네트워크에서 얻을 수 있는 q value를 평균을 내어 action을 선택하는 DQN이다. 이때 모든 agent들이 동일한 mini-batch 순서에 따라 학습된다는 점이 특징이다. Ensenble-DQN의 loss function은 다음과 같다.

$$
L(\theta) = {1 \over K} \Sigma_{k=1}^K E_{s,a,r,s' \backsim D} [ l_\delta ( Q_\theta^k(s,a) -r -\gamma \max_{a'} Q_{\theta'}^k (s', a') ) ]
$$

여기서 $$l_\delta$$는 Huber Loss이다.

#### REM Algorithm

앙상블 모델에서는 agent의 갯수가 많아지면 많아질수록 성능이 높아지는 경향이 있다고 한다. 이러한 점을 고려해 볼 때 많은 agent들을 효율적으로 학습시키는 방법 또한 중요하다고 할 수 있는데, REM은 이를 drop out과 유사한 방법으로 해결하려고 하는 알고리즘이다.

우선 REM 또한 복수의 agent를 사용한다는 점은 Ensenble-DQN과 동일하다. 하지만 모든 agent가 동일하게 학습되는 Ensenble-DQN과는 달리 REM은 매 학습마다 업데이트되는 agent가 다르고, 그 크기 또한 서로 다르다. 즉 n개의 agent가 있다면, 각각이 학습될 확률을 $$\alpha$$라는 random drawed categorical distribution으로 매 step마다 정하고 그 크기에 맞춰 loss를 구하게 된다. 정확한 REM의 Loss function은 다음과 같다.

$$
L(\theta) = E_{s,a,r,s' \backsim D} [ E_{\alpha_1, ... \alpha_K \backsim P_\Delta} [ l_\delta ( \Sigma_k \alpha_k Q_\theta^k (s,a) - r - \gamma \max_{a'} \Sigma_k \alpha_k Q_{\theta'}^k (s',a') ) ] ]
$$

여기서 $$P_\Delta$$는

$$
\begin{multline}
standard \ (K-1) \ simplex \ \Delta^{K-1}\\
= \{ \alpha \in \rm I\!R : \alpha_1 + \alpha_2 + \alpha_3 + ... + \alpha_K = 1, \alpha_k \geqq 0, k = 1, ..., K \}
\end{multline}
$$

의 확률 분포를 의미한다.

논문에서는 실험을 위해 가장 기본적인 $$P_\Delta$$를 사용했다고 하는데, 구체적으로는 각 $$\alpha_k$$ 값을 구하기 위해 우선 $$\alpha' \backsim U((0,1)$$ 구한 뒤 $$ \alpha_k = \alpha'_k / \Sigma \alpha'_i $$로 정했다고 한다.

#### performance of REM

논문에서는 REM과 함께 pure DQN, Ensenble-DQN, Bootstrapped-DQN 그리고 distributional RL의 한 종류인 QR-DQN에 대해 60개의 Atari game 환경에서 Online, Offline 실험했고, 그 결과 Online 환경에서는 QR-DQN이 REM에 비해 약간 더 좋지만, Offline 환경에서는 REM이 가장 좋은 성능을 보였다고 한다.
