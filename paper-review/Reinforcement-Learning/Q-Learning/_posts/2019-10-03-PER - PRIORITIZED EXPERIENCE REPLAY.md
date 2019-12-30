---
layout: post
title: PER) PRIORITIZED EXPERIENCE REPLAY
---

# 논문 제목 : PRIORITIZED EXPERIENCE REPLAY

- David silver 등
- 2016
- <https://arxiv.org/abs/1511.05952>
- 2019.10.03 정리

## 세 줄 요약

- Prioritized experience replay란 TD error가 큰 것에 우선 순위를 부여하여 보다 학습이 자주 이뤄지도록 하는 방법이다.
- replay memory에서 TD error가 큰 순서대로만 뽑는(greedy) 방법은 단점이 분명하며, TD error에 비례하여 뽑힐 확률이 높아지는(stochastic) 방법이 더 낫다.
- random sampling이 아니므로 전체 replay memory의 sample의 분포에 비해 bias가 발생하는데 이로 발생하는 문제는 Importance Sampling을 학습에 적용하여 해결한다.

## 내용 정리

### prioritized experience replay

- replay memory에 새로운 방법을 적용한다고 한다면 (1) 어떤 경험을 저장할 것인지, (2) 저장된 경험 중 어떤 경험을 추출할 것인지 두 가지에 대해 생각해 볼 수 있다.
- prioritized experience replay는 (2) 에 관한 방법론이다. 즉 replay memory에 저장되는 경험은 기존의 experience replay와 마찬가지로 FIFO를 따른다.

#### 경험의 우선순위와 관련된 선행 연구

- 설치류의 두뇌에 대한 연구 중 reward와 관련이 높은 sequence of prior experience가 보다 빈번하게 학습되는 것이 확인되었다.
- 최근 연구 중 positive-reward, negative-reward의 두 개의 experience memory를 만들어서 사용하는 방법이 제시되었다.
  - 성능이 약간 높아졌다고 하지만 두 보상이 확실하게 구분되는 영역에서만 사용될 수 있다는 것이 한계로 지적된다.

#### 각 경험의 중요도 평가

- prioritized experience replay에서 가장 중요한 것은 각 경험의 중요도(importance of each transition)를 판단하는 기준 또는 방법이다.
- 기준으로 삼을 만한 것으로 TD error가 있다.

  `TD error = R𝗍₊₁ + γV(S𝗍₊₁) - V(S𝗍)`

- 즉, TD error란 '새롭게 업데이트 된 V(s)의 값과 기존 V(s) 값의 차이' 라고 할 수 있다.
  - 이는 곧 현재의 value function에 있어 어떤 경험이 얼마나 충격적인지에 대한 지표로도 볼 수 있다.
  - "How surprising or unexpected the transition is: specifically, how for the value is from its next - step bootstrap estimate"

### experience replay 방법 간 비교

#### 1. experience replay(uniform sampling)

- 기존의 experience replay 에서는 replay memory에 experience를 저장하고 이를 random 추출하여 학습 대상을 정하였다.
- 이를 통해 다음과 같은 장점을 얻을 수 있었다.
  - data를 한 번만 학습하고 버리는 것이 아니라 중복 학습이 가능해졌다.
  - 시간 순서와 무관한 학습이 가능해졌다.
- 하지만 다음과 같은 단점을 가지고 있었다.
  - 학습의 양이 크게 증가하여 자원이 낭비된다는 문제가 있었다.
- prioritized experience replay는 보다 중요한 experience를 많이 학습하는 방식으로 효율성을 높이는 방법이라고 할 수 있다.

#### 2. Greedy prioritization

- TD error를 기준으로 greedy TD-error prioritization algorithm을 만들 수 있다.
  - "The transition with the largest absolute TD error is replayed from the memory"
- 즉 TD error가 가장 높은 경험 순으로 추출하여 학습하는 것이다.
- 하지만 이 방법은 다음과 같은 문제를 가진다.
  - 전체 replay memory에 저장된 경험들 중에서 일부 경험만 반복적으로 학습할 가능성이 있다.
    - 처음 경험한 것이라 하더라도 TD error가 작으면 학습되지 않는다.
  - noise에 취약하다. 즉 noise로 인해 TD error가 높게 나오더라도 이를 반복 학습하게 된다.
  - 소수의 경험만을 반복 학습하면 over fitting 가능성이 높아진다.
    - "The lack of diversity that makes the system prone to over-fitting"
- 이러한 문제를 해결하기 위해 논문에서는 uniform sampling과 greedy prioritization의 중간 수준인 stochastic sampling method를 제안하고 있다.

#### 3. Stochastic sampling method

- TD error를 기준으로 하되 높은 순서대로 뽑는 것이 아니라 비례하여 뽑힐 확률이 높아지도록 하는 방법이다.
- 어떤 한 경험이 뽑힐 확률은 다음과 같이 나타낼 수 있다.

  `P(i) = p𝚒ᵅ / Σ𝗄 p𝗄ᵅ`

- p𝚒 는 i 번째 sample이 가지는 priority이다.
- a는 prioritization이 얼마나 적용되는지 나타내는 hyper parameter이다. a = 0 일 경우 uniform sampling과 동일해진다.
- 이때 p𝚒 의 정의로 논문에서는 proportional prioritization과 rank-based prioritization 두 가지 방법을 제시하고 있다.

##### proportional prioritization

- TD error의 절대값을 곧바로 p𝚒 로 이용하는 방법이다.

  `p𝚒 = |δ| + ε`

- δ는 TD error를 의미하고, ε은 매우 작은 양수로 pi가 0이 되는 것을 방지한다.

##### rank-based prioritization

- replay memory 내에 있는 sample 들의 TD error 크기(`|δ|`)의 순서에 따라 priority를 부여하는 방법이다.

  `p𝚒 = 1 / rank(i)`

### sampling에 따른 bias 문제 : Importance Sampling

- random sampling이 아닌 sampling 방법의 경우 전체 replay memory의 분포와 sampling 된 결과들의 분포에 차이가 있다는 것이다
- 이러한 문제를 해결하기 위해 Importance-Sampling(IS) wᵢ를 도입한다.
- 기존 DQN에서는 학습이 δᵢ를 기준으로 이뤄졌다면 여기서는 wᵢδᵢ를 기준으로 이뤄진다.
- wᵢ의 의미는 다음과 같다.

  `wᵢ = ( (1/N) * (1/P(i)) )ᵇ`

- 여기서 N은 replay memory의 크기를 뜻한다.
- b는 0에서 1로 epoch이 진행될 때마다 점점 커진다.

- 전형적인 강화학습의 학습 과정을 살펴보면 학습의 마지막 과정에서 sample의 분포에 따른 bias가 미치는 영향이 크다고 한다.
  - "In typical reinforcement learning scenarios, the unbiased nature of the updates is most important near convergence at the end of training"
- 이러한 점을 고려하여 학습이 진행됨에 따라 bias를 줄이기 위한 wi의 값을 늘려나가게 된다(b가 0에서 1로 점점 커지는 이유).
- initial b 값의 경우 0보다 크고 1보다 작은 어떤 수로 설정할 수 있으며, initial b에서 1로 커지는 속도 또한 하이퍼 파라미터로 설정할 수 있다. 학습의 마지막 epoch에서 b값이 1이 되면 된다.
