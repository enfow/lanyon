---
layout: post
title: HER) Hindsight Experience Replay
---

# 논문 제목 : Hindsight Experience Replay

- Marcin Andrychowicz 등(OpenAI)
- 2018
- <https://arxiv.org/abs/1707.01495>
- 2019.10.04 정리

## 세 줄 요약

- action space가 너무 큰 경우 일반적인 강화학습 알고리즘으로는 학습이 되지 않는 문제가 있다. 이는 거의 대부분의 reward가 음수로 나오기 때문이다.
- episode의 모든 state를 goal state로 설정하여 복수의 goal에 대한 학습을 진행하여 이러한 문제를 해결할 수 있는데, 이를 Hindsight experience replay(HER)라고 한다.
- multiple goal을 설정하는 방법은 UVFA의 방법론을 사용했다. 즉 state에 각 goal을 concatenation해 입력으로 집어넣는다(구체적인 방법은 달리 할 수도 있는 것 같다).

## 내용 정리

### Background

- 일반적인 강화학습 알고리즘은 성공한 경험(reward > 0)을 대상으로만 학습을 진행한다. 하지만 사람의 경우 좋지 않은 결과(achieving undesired outcome)에 대해서도 학습한다.
  - 사람이 하키 게임을 할 때 골키퍼에게 슈팅이 막힌 상황을 생각해보자. 이때 사람은 더 강하게 치거나 혹은 조금 더 깊숙한 방향으로 쳐야겠다고 학습한다.
- HER 은 이러한 실패를 통한 학습에서 출발한다.

#### UVFA(Universal Value Function Approximator)

- UVFA는 DQN을 토대로 한 개 이상의 goal이 존재하는 상황을 위해 만들어진 방법이다.
- UVFA에서는 기본적으로 복수의 goal을 가정하며 이러한 목표는 action과 reward 모두에 영향을 미친다.
  - 즉, policy 𝝅 는 state 뿐만 아니라 현재의 goal 을 함께 입력으로 받아 action을 반환하고, reward 또한 state, action, goal 세 개의 조합으로 결정된다.
  - Q function은 다음과 같이 정의된다.

  `Q = (s𝗍, a𝗍, g) = E[R𝗍 | s𝗍, a𝗍, g]`

- "Hindsight Experience Replay which allows sample-efficient learning from rewards which are sparse and binary and therefore avoid the need for complicated reward engineering"

#### motivated example

- Bit-flipping 게임에서 선택할 수 있는 정수의 개수가 40 이상을 넘어가면 일반적인 DQN으로는 학습이 불가능하다. 즉, positive reward를 받을 가능성이 없다(never).
  - count-based exploration(Ostrovski), boostrapped DQM(Osband) 등 발전된 탐색 방법들이 있지만 이와 같은 문제에서는 도움이 되지 않는다.
- 지금까지는 이러한 문제를 해결하기 위해서 Reward function을 적절하게 조정하는 방법을 사용했다. Bit-flipping 문제도 이로 해결이 가능하다.
- 하지만 문제가 보다 복잡해지면 reward function을 일일이 조작하는 것이 쉽지 않다. 특히 domain knowledge를 요구하는 분야의 경우 더 어려워진다.
- HER는 이러한 문제를 해결하는 방법이라고 할 수 있다.

### HER : Hidesight Experience Replay

#### 아이디어

- HER의 기본적인 아이디어는 한 episode의 최종 state에 도달하는 방법을 학습하는 것이다. 이를 통해 이후 episode 진행 시 탐색의 범위가 보다 넓어지고 최종적으로는 기존에는 도달하지 못한 목표에 도달할 수 있다는 것이다.
- HER에서는 하나의 episode가 끝나게 되면 episode의 경로(trajectory)에 대해 각각 다른 goal을 가지고 학습하게 된다.
- 즉 episode 상의 각 state가 goal이 되어 학습이 이뤄지므로 항상 reward가 -1이 되어 episode에 대해 학습이 이뤄지지 않는 상황은 발생하지 않게 된다.
- 또한 결과적으로 episode의 최종 state까지 도달하는 방법을 학습하게 되므로 다음 episode에서는 이전 episode 들의 최종 state에까지 보다 쉽게 도달하게 되어 결과적으로 탐색의 범위가 늘어나게 된다.

#### multi-goal을 적용하기 위한 구체적인 방법

- 기본적으로 UVFA를 사용한다. 즉 goal이 복수(multiple-goal)로 존재하며 action을 구하기 위해 현재 state와 현재 goal을 입력한다.
- episode가 s0, s1, ... , s𝑡 로 구성되어 있다고 하자. 이때 HER는 initial state s0에서 s𝗍로 가는 것을 학습하려한다.
- epsiode에서는 sample을 다음과 같이 t-1개 추출할 수 있다.
  - (s₀, a₀, r₀, s₁), (s₁, a₁, r₁, s₂), (s₂, a₂, r₂, s₃) ... , (s𝗍₋₁, a𝗍₋₁, r𝗍₋₁, s𝗍)
- 첫 번째는 다음과 같이 최종 목표 g에 대해 state와 goal을 concatenation 한 값을 replay memory에 저장한다. 이는 standard experience replay와 동일하다.
  - 예를 들어 `(s𝗍₋₁||g, a𝗍₋₁, r𝗍₋₁, s𝗍 || g)` 와 같은 식이다.
- 다음으로는 각 sample에 대해 episode 상의 모든 state를 goal로 설정하여 concatenation 한 값을 replay memory에 저장한다.

  ```sudo
  for t=1, T-1 do
     (s₀||g', a₀, r₀, s₁||g')
     store (s₀||g', a₀, r₀, s₁||g') in replay memory
  end for
  ```

- 이후 minibatch 를 구성하고 학습을 진행하는 것은 기본적인 방식과 크게 다르지 않다.
