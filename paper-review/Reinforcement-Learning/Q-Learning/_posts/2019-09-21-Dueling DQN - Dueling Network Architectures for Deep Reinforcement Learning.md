---
layout: post
title: Dueling DQN) Dueling Network Architectures for Deep Reinforcement Learning
---

# 논문 제목 : Dueling Network Architectures for Deep Reinforcement Learning

- Ziyu Wang 등
- 2015
- <https://arxiv.org/abs/1511.06581>
- 2019.09.21 정리

## 세 줄 요약

- Dueling architecture란 모든 action에 대해 직접적으로 Q value를 구하는 것이 아니라 value function V와 advantage function A의 합으로 Q value를 구하는 구조를 말한다.
- 이와 같이 두 개로 나누는 것을 Advantage updating 이라고 하며 이를 통해 빠르고 정확한 수렴이 가능하다.
- 성능을 높이기 위해 Prioritized replay를 사용했다.

## 내용 정리

### Deuling architecture

- 많은 state에서 가능한 모든 경우의 action을 계산할 필요가 없다.
  - 예를 들어 Enduro game 에서는 충돌 직전에만 왼쪽 또는 오른쪽으로 움직이면 된다. 하지만 많은 state에서는 어떠한 action을 취하던 간에 충돌이 일어나지 않는다.
  - 반면 어떠한 행동을 선택하더라도 충돌이 발생하는 경우도 존재한다.
- 이러한 점을 고려해 볼 때 모든 state에서 모든 action의 value를 구하고 그에 따라 행동하는 것보다는 좋지 않은 state에 처하지 않도록 하는 것이 더 중요할 수 있다.
- 이러한 점에서 각 state의 value function을 감안하여 action을 선택하는 것이 더 나은 의사결정으로 이어질 수 있다.

### Deuling architecture의 구조적 특성

- Dueling architecture란 state value representation과 action advantage representation의 합으로 Q-function의 값을 구하는 구조를 말한다.
  - "explicitly seperated the represenation of state values and (state-dependent) action adcantage"
- 두 가지 represenation은 하나의 CNN module을 공유한다.
- CNN module 이후 나누어진 두 개의 represenation은 특정한 지정(supervision)없이 자동적으로 state value function과 advantage function의 추정치를 만들게된다. 그리고 특수한 결합 레이어(special aggregating layer)에 의해 합쳐져 state-action value Q-function을 만들게 된다.

- V function의 값은 scalar 값이 되고, A function의 값은 '현재 state에서 취할 수 있는 모든 action들의 수'의 크기를 가지는 vector값이 된다. 그리고 최종 Q function의 값은 vector의 각 element에 scalar 값을 더하여 구하게 된다.
  - `Q(s, a; θ, α, β) = V (s; θ, β) + A(s, a; θ, α)`
- 여기서 문제 중 하나는 더해진 Q로는 V와 A의 값을 recover 하기 어렵다는 점이다.
  - "unidentifiable in the sense athe given Q we cannot recover V and A uniquely"
  - 이는 적절한 학습을 방해하는 요인이 되며 결과적으로 performance에 부정적인 영향을 준다.
- 이러한 문제를 해결하기 위해 A 에 max(A)의 값을 빼주어 A의 모든 element가 0 또는 음수의 값을 가지도록 하는 방법이 있다.
  - `Q(s, a; θ, α, β) = V (s; θ, β) + (A(s,a;θ,α)− max A(s,a′;θ,α))`
  - `Q(s, a*; θ, α, β) = V (s; θ, β)` 가 성립하게 되어 V 값을 판단할 수 있게 된다.
- 안정적인 optimization을 위해 max A의 값이 아닌 A의 평균값을 빼주는 것도 가능하다. 논문에서는 이 방법을 사용했다.

### Advantage updating

- value function과 advantage function 두 개로 나누는 아이디어는 1993년 Baird에 의해 처음 사용되었다.
  - 일반적인 Q-learning보다 빠른 수렴 속도가 특징이다.
- Advantage function은 다음과 같이 정의된다.
  - `Q𝝅(s, a) = E𝒔'[r + 𝛄 max𝖺' Q*(s',a')| s, a]`
  - `A𝝅(s, a) = Q𝝅(s, a) - V𝝅(s)`
- 즉 Advantage function A(s, a) 는 Q function 값에 V function 값을 뺀 것이다. 이러한 점에서 A(s, a)는 각 action의 상대적인 가치를 나타낸대고 할 수 있다.
  - "The advantage function subtracts the state value from Q to obtain a relative measure of the importance of each action"
- `E𝖺～𝝅(𝑠)[A𝝅(s,a)] = 0` 이 성립한다.

### Prioritized replay

- "The key idea was to increase the replay probability of experience tuples that have a high expected learning process"
- priority replay를 통해 DDQN 등 여러 RL 기법들이 SOTA를 찍었다.
- <https://arxiv.org/abs/1511.05952>

## reference

- <https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/>
