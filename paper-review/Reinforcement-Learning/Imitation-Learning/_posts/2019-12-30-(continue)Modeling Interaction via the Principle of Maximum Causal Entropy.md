---
layout: post
title: Maximum Entropy) Modeling Interaction via the Principle of Maximum Causal Entropy
---

# 논문 제목 : Modeling Interaction via the Principle of Maximum Causal Entropy

- Brian D. Ziebart 등
- 2010
- <http://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf>
- 2019.12.30 정리

## 세 줄 요약

## 내용 정리

Maximum entropy의 원칙(Jaynes1957)은 통계적 모델을 설계하는데 있어 이론적인 바탕이 되는 역할을 했다. 여기에 조건(conditional)을 더하여 sequence of side information 을 고려하도록 한 모델 또한 다양한 분야에 활용되어왔다. 여기서 말하는 side information 이란 예측의 대상이 되지는 않지만, 예측을 하는 데에 관련이 있는 변수를 말한다.

논문에서는 side information이 동적이고, 확률적인 과정에 따라 상호작용하게 되는 환경에서의 조건부 확률 분포(conditional probability distribution) 에 maximum entropy를 적용하는 방법에 대해 이야기한다. 이러한 환경의 대표적인 예시가 확률적인 환경과 상호작용하는 agent 이다. agent는 현 시점의 state와 action만이 주어진 상황에서 미래 states 의 분포를 모델링해야 한다. 하지만 확률성 때문에 어떠한 state 로 넘어갈 것인지 확실하게 알 수 없다. 다시말해 미래의 state는 이전의 action에 대해 causal influence를 가지지 않는다. 기존에 많이 사용되어온 Conditional maximum entropy 방법을 이와 같은 상황에 곧바로 적용하는 것은 적합하지 않다. 이러한 문제를 해결하기 위해 principle of maximum causal entropy를 제시한다.

notation

```
A : sequence of action variable
S : sequence of state variable
```

the probability of A causally conditioned on S

```
P(Aᵀ||Sᵀ) ≜ 𝚷ᵀP(A𝗍 | S₁﹕𝗍, A₁﹕𝗍₋₁)
```

conditional probability

```
P(A|S) ≜ 𝚷ᵀP(A𝗍 | S₁﹕𝖳, A₁﹕𝖳₋₁)
```

위의 두 식을 비교해보면 결국 S₁﹕𝗍 와 S₁﹕𝖳 에 있다. 즉, causally conditional probability 는 전체 state 가 아닌 부분, S₁﹕𝗍 만을 조건으로 받는다는 차이가 있다. 이는 causal entropy 와 연관된다.

causal entropy

```
H(Aᵀ||Sᵀ) ≜ ΣᵀH(A𝗍 | S₁﹕𝗍, A₁﹕𝗍₋₁)
```
