---
layout: post
title: GAIL) Generative Adversarial Imitation Learning
---

# 논문 제목 : Generative Adversarial Imitation Learning

- Jonathan Ho 등
- 2016
- [논문](<https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf>)
- 2019.12.30 정리

## 세 줄 요약

## 내용 정리

### Imitaion Learning

imitation learning은 주어진 전문가의 행동(expert demonstrations)을 모사하는 방식으로 policy를 학습하는 방법이다. 일반적으로 imitation learning은 전문가의 행동으로 이뤄진 데이터셋을 대상으로 학습을 진행하며, 그 과정에서 데이터셋 이외의 새로운 데이터를 받을 수 없다. 즉 offline learnning을 기본으로 한다.

대표적인 방법론으로는 **Behavior cloning**과 **Inverse reinforcement learning(IRL**)이 있다. Behavior cloning 은 state-action pair 로 표현된 전문가들의 행동을 policy가 그대로 묘사하는 방법이다. IRL은 전문가들의 행동을 이용하여 성능이 가장 좋은 cost function 을 찾아내는 방법이다.

Behavior cloning은 단순해 보이지만 covariate shift 에 의한 compounding error 때문에 많은 양의 데이터를 필요로 한다는 문제가 있다. 반면 IRL 은 전체 trajectory 를 다른 것보다 우선시하도록 cost function을 학습하기 때문에 compounding error에 있어 자유롭다는 장점이 있다. 실제로 택시 운전사의 행동 예측 등에 있어 IRL은 성공적인 결과를 보이기도 했다.

하지만 IRL 또한 여러가지 문제점을 가지고 있는데, 첫 번째는 학습에 비용이 많이 든다는 것이고 두 번째는 어떤 action을 선택할지를 학습하는 것이 아니라 cost function을 학습한다는 것이었다. GAIL은 IRL의 이러한 문제점들을 해결하려는 데에서 출발한다.

### IRL

논문에서 사용되는 기본적인 notation은 다음과 같다.

- $$ S $$, $$ A $$ : 유한한 state, action의 집합
- $$\pi$$ : 선택 가능한 모든 policy의 집합(𝝅 ∈ 𝚷)
- $$E_\pi$$ : expert policy

policy 𝝅 가 만들어내는 trajectory를 다음과 같이 기대값 형태로 나타낼 수 있다.

$$ {E}_{𝝅}[c(s,a)] ≜ {E}[\sum_{t=0}^\infty \gamma^t c(s_t, a_t)] $$

본 논문에서는 maximum causal entropy IRL 의 해가 있다고 가정한다. 즉, 아래 수식의 해가 있다고 가정하는 것이다.

$$ IRL(\pi_E) = maximize_{c \in C} \left(\vcenter{\min_{\pi \in \Pi} - H(\pi) + E_{\pi}[c(s,a)] }\right) - E_{\pi_{E}}[c(s,a)] $$

policy 𝝅 의 감가된 causal entropy를 의미하는 H(𝝅) 는 다음과 같이 정의된다.

$$ H(\pi) ≜ E_{\pi}[-\log\pi(a|s)] $$

여기서 c 는 cost function을 의미하고 $$\pi_{E}$$ 는 expert policy를 뜻하는데, expert policy는 일반적으로 샘플링된 복수의 trajectory 형태로 제공되기 때문에 데이터셋으로부터 추정하게 된다.

Maximum causal entropy IRL의 식을 보면 expert policy에 대해서는 낮은 cost를, 그 외의 다른 policy에 대해서는 높은 cost를 기대한다. 즉 우변의 앞 부분인 $$ \left(\vcenter{\min_{\pi \in \Pi} - H(\pi) + E_{\pi}[c(s,a)] }\right) $$ 는 최대한 커지길 기대하고, 뒷부분인 $$ E_{\pi_{E}[c(s,a)]} $$ 는 작아지길 바란다고 해석할 수 있다.

이때 최적의 policy를 선택하는 부분(Reinforcement Learning part), 즉 우변의 앞 부분만을 보게 되면 다음과 같이 수식을 정리할 수 있다.

$$ RL(c) = argmin_{\pi\in\Pi} - H(\pi) + E_\pi[c(s,a)] $$

이에 따라 policy는 높은 엔트로피를 유지하면서도 기대 누적 cost를 최대로 하게 된다.

기본적으로 GAIL 은 IRL의 방법론을 이용하면서도 어떻게 하면 직접 action을 알려주는 policy 를 학습시킬 수 있을지 하는 고민에서 출발한다.

### regularized IRL

집합 $$C$$ 를 강화학습에 의해 학습 가능한 cost function 의 집합이라고 하자. cost function은 복잡한 전문가의 행동을 표현하기 위해 가우시안 프로세스 또는 뉴럴 네트워크를 사용한다.

$$C = \rm I\!R^{SxA} = \{c:S x A \rightarrow \rm I\!R\} $$

이때 $$C$$의 크기가 크기 때문에 유한한 데이터셋에 대해서 쉽게 오버피팅되는 문제가 있다. 따라서 이러한 문제를 줄이기 위해 논문에서는 convex cost function regularizer를 도입하고, 이를 $$ \psi $$ 로 표기한다. **$$\psi$$ regularized IRL**은 다음과 같다.

$$ IRL_{\psi}(\pi_E) = argmax_{C\in \rm I\!R^{SxA}} - \psi(c) + \left(\vcenter{\min_{\pi \in \Pi} - H(\pi) + E_{\pi}[c(s,a)] }\right) - E_{\pi_{E}[c(s,a)]} $$

IRL 식이 새롭게 정의되면서 이에 따라 구해지는 policy, 즉 RL 식도 다시 구할 수 있다. 이를 표현하기 위해 $$IRL_{\psi}(\pi_E)$$ 에 의해 구해진 cost function 을 $$ \tilde c$$ 라고 하자. 그럼 위에서 정의된 RL 를 이용해 $$\tilde c$$ 에 맞는 **policy RL($$ \tilde c$$)** 를 구할 수 있다.

### Characterized **policy RL($$ \tilde c$$)**

policy RL($$ \tilde c$$) 를 보다 구체화하기 전에 논문에서는 두 가지 개념을 요구한다.

#### 1. distribution of state-action pair : $$p_\pi$$

첫 번째는 distribution of state-action pair에 관한 것으로, bellman equation에서 $$p_\pi$$ 로 표기되는 것이다. 이를 어떤 정책 $$\pi$$의 occupancy measure 로 이용할 것이라고 한다. 보다 구체적으로는 아래와 같이 cost function의 기대값을 구할 때 사용한다.

$$E_\pi[c(s,a)] = \sum_{s,a}p_\pi(s,a)c(s,a)$$

#### 2. convex conjugate

두 번째는 convex conjugate에 대한 내용이다. 최적화와 관련된 내용 같은데([링크](<https://wikidocs.net/17428>)) 이 부분은 공부가 필요해 생략하도록 한다. 논문에 나와있는 수식은 다음과 같다.

for function $$f : \rm I\!R^{SxA} \rightarrow \bar{\rm I\!R}$$

convex conjugate $$f^* : \rm I\!R^{SxA} \rightarrow \bar{\rm I\!R}$$

이때 $$f^*(x) = \sup_{y \in \rm I\!R^{SxA} }x^Ty - f(y)$$ 로 정의된다.

위의 두 가지 내용을 전제로 하여 논문에서는 아래의 proposition 과 lemma 들을 통해 **policy RL($$ \tilde c$$)** 를 설명하고 있다.

#### proposition 3.1

$$RL \circ IRL_{\psi}(\pi_E) = argmin_{\pi \in \Pi} - H(\pi) + {\psi}^*(p_\pi - p_{\pi_E}) $$

이는 regularized IRL 이 expert의 policy와 occupancy measure 면에서 $$\psi^*$$를 기준으로 가까운 policy를 찾는 것이다.

#### proposition 3.2

$$
\begin{multline}
\shoveleft{Suppose \ p_{\pi_E} > 0} \\
\shoveleft{If \ \psi \ is \ a \ constanct \ function, \bar c \in IRL_{\psi(\pi_E)}, \ and \ \tilde \pi \in RL(\tilde c),} \\
then \ p_{\tilde \pi} = p_{\pi_E} \\
\end{multline}
$$

## Additional study

- Covariate Shift
  - compounding error와 covariate shift의 관계
- Lagrangian
  - dual optimal
- gaussinan process
- convex conjugate
- psi
