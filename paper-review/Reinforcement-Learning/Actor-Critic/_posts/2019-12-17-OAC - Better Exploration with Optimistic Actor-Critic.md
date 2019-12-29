---
layout: post
title: OAC) Better Exploration with Optimistic Actor-Critic
---

# 논문 제목 : Better Exploration with Optimistic Actor-Critic

- Kamil Ciosek, Quan Vuong 등
- 2019
- <https://arxiv.org/abs/1910.12807>
- 2019.12.17 정리

## 세 줄 요약

- 2019 NIPS에서 발표된 model free reinforcement 의 새로운 SOTA 모델이다.
- SAC, TD3과 같은 기존 SOTA 모델은 어느 정도 학습이 이뤄지면 특정 영역에 대해 반복 탐색하게 되는 경우가 많다.
- OAC에서는 exploration policy를 별도로 두어 이러한 문제를 해결하려고 한다. 이 때 아직 탐색하지 못한 영역을 찾기 위해 q function의 upper bound 를 도입하고 이를 계산하는 방법을 제시한다. 그리고 target policy와 exploration policy 간의 차이를 조정하기 위해 KLD 제약 조건을 둔다.

## 내용 정리

### Inefficiency of existing algorithms

논문은 SAC, TD3를 비롯한 지금까지의 강화학습 알고리즘들에서 사용하고 있는 탐색 방법이 비효율적이라고 주장하며, 구체적인 이유로 "pessimistic underexploration", "directional uniformedness" 두 가지를 제시한다.

#### pessimistic underexploration

Actor-Critic과 같이 q value를 이용한 강화학습 알고리즘의 오랜 문제 중 하나는 q value가 overestimate 되기 쉽다는 것이다. 이러한 문제를 해결하기 위해 SAC와 TD3에서는 두 개의 critic을 만들고 두 네트워크의 출력값 중 보다 작은 것을 사용하는데, 이를 lower bound approximation 이라고 한다. 하지만 q value의 값이 여전히 부정확한 상황이라면 lower bound를 사용하는 방법 또한 여전히 위험(very harmful)하다.

논문의 figure 1a에서는 실제 q value와 lower bound q value 간의 값을 비교하며 그 위험성을 보여주고 있다. 여기서 말하는 위험성이란 q value가 높아질 가능성이 있는 영역이 있음에도 불구하고 lower bound q value가 최대화되는 지점에서 멈추고 그 주변만 탐색하게 된다는 것이다. 현재 continuous action space에 대응하는 강화학습 알고리즘에서 자주 사용되는 OU noise를 이용한 exploration 방법과 함께 보면 이러한 문제는 더욱 확실해진다. 즉, 어느 정도 탐색이 끝나 lower bound q value가 maximum이 되는 값에 도달하면 이를 평균으로 하는 좁은 영역만 반복적으로 탐색하게 된다는 것이다. 논문에서는 이러한 문제를 pessimistic underexploration이라고 한다.

#### directional uniformedness

gaussian policy를 이용하여 탐색을 하는 경우 평균을 중심으로 정방향과 역방향의 action들이 모두 둥일한 확률분포를 가지고 있다. 하지만 gradient algorithm에 따라 점진적으로 발전하는 policy network의 특성상 과거에 지나온 영역과 그렇지 않은 영역을 동일한 수준으로 exploration하는 것은 비효율적이다. 논문에서는 이를 두고 "Exploration in both directions is wasteful"이라고 언급한다.

### Better Exploration with Optimism

#### Exploration policy

위에서 제시한 기존 강화학습 알고리즘들의 문제들은 exploration과 관련된 것으로, 이를 해결하기 위해 OAC에서는 exploration policy를 별도로 두고 있다.

exploration policy는 upper bound q value와 KL Divergence를 이용한다. 간단히 설명하자면 upper bound q value를 이용해 지금까지 탐색하지 못한 영역에 접근할 수 있도록 하되, KL divergence로 제약을 두어 exploration policy에 따른 탐색 영역이 target policy의 결과와 크게 달라지는 것을 방지한다.

#### upper bound

논문에서는 uppder confident bound에 근사하기 위해 세 단계를 거친다고 한다. 첫 번째는 Q의 평균과 표준편차를 구하는 것이고, 두 번째는 Q의 평균과 표준편차를 이용해 expected upper bound를 찾는 것이다. 세 번째는 다시 이 expected value를 linear approximation하여 uppder confident bound를 찾는 것이다.

첫 번째, Q의 평균과 표준편차를 구하는 것에 있어 평균은 어렵지 않다. 두 개의 critic Q value를 합하여 나누어주면 되기 때문이다. 하지만 표준편차를 구하기 위해서 논문에서는 다음과 같은 방식을 도입하고 있다. (appendix c 부분 추가하기)

#### KL Divergence constraint

q function의 upper bound 만 maximize 하는 방향으로 exploration이 이뤄지면 실제 target policy와 exploration policy 간의 차이가 매우 커질 수도 있다. 이 경우 update의 안정성에 문제가 생길 수 있어, target policy와 exploration policy 간의 차이를 일정 수준 이내로 조정할 필요가 있다. 이러한 문제를 해결하기 위해 OAC에서는 두 gaussian distribution 간의 KLD를 일정 수준 이내로 제약 조건을 도입하고 있다.