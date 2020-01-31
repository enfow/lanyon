---
layout: post
title: Off-Policy Actor-Critic
category_num: 0
---

# 논문 제목 : Off-Policy Actor-Critic

- Thomas Degrix, Marth White, Richard Sutton
- 2013
- [논문 링크](<https://arxiv.org/pdf/1205.4839.pdf>)
- 2019.11.07 정리

## 세 줄 요약

- 처음으로 Actor-Critic 알고리즘에 off-policy를 적용한 논문이다.

## 내용 정리

### On-policy & Off-policy

on-policy와 off-policy는 agent의 학습 방법에 관한 차이로, on-policy는 현재 policy의 행동에 의해서 결정된 transaction으로만 학습을 하는 것이고, off-policy는 그렇지 않은 경우를 말한다. 지금까지 online setting, 즉 환경과 agent가 실시간으로 데이터를 주고 받는 상황에서는 on-policy 방법으로만 수렴성이 보장되었고 off-policy로는 학습에 어려움이 많았다.

하지만 off-policy는 상당히 많은 장점을 가지고 있고 적용할 가능성 또한 넓어서 이를 이용한 학습을 다양하게 시도해왔다. 대표적인 off-policy의 장점으로 논문에서는 다음 세 가지를 언급하고 있다.

1. 탐색을 하는 도중에도 optimal policy에 대해 학습할 수 있다.
2. demonstration에 대해 학습할 수 있다.
3. 병렬적인 복수의 작업을 수행하며 학습을 진행할 수 있다.

#### off policy의 수렴성 문제

대표적인 off-policy 알고리즘이 Q-learning(Watkins&Dayan 1992)인데, 이 경우 근사를 이용하지 않는 tabular의 경우 수렴하지만 선형 근사를 사용하는 경우에는 발산할 수 있다는 문제(Baird)를 가지고 있다. 이러한 문제를 해결하는 알고리즘으로 Least-squares 계열의 LSTD, LSPI 등이 있었지만, 선형 근사를 이루기 위해 많은 처리 비용이 든다는 점에서 좋은 해결 방법은 아니었다고 지적한다. 최근에는 gradient-TD 방법을 사용하는 Greedy GQ 등의 알고리즘이 제시되었다고 한다.

하지만 Greedy GQ와 같은 action-value 방법론들 또한 문제가 있는데 논문에서는 다음 세 가지를 제시한다.

1. action-value 방법론들은 deterministic target polies를 가지고 있다. 하지만 많은 문제들이 확률적인 optimal policies를 가지는데 대표적으로 adversarial setting, partially observable MDP 등이 있다.
2. greedy action을 action value function에 따라 찾는 방법은 action space가 클 경우에는 좋은 방법이 아니다.
3. action value function의 변화에 매우 민감하다. 이 경우 실제 문제를 해결하기에는 위험성이 크다.

이러한 문제를 해결하는 방법 중 하나는 PG 계열의 알고리즘인 actor-critic을 사용하는 것이다. 실제로 on-poicy PG 계열의 actor-critic은 연속적인 action space를 갖는 환경에서 많은 문제를 성공적으로 해결할 수 있음을 보였다. 그리고 본 논문에서는 여기서 나아가 off-policy를 적용한 actor-critic 모델을 제시하고 있다.

### Problem Setting

#### 1. value function $$V^{\pi, \gamma}(s)$$

$$\pi : S X A \rightarrow (0,1]$$에 대한 value finction은 다음과 같이 정의된다. 이때 시작 시점은 $$t$$이고, 마지막 시점은 $$t+T$$인데 이는 $$\gamma: S \rightarrow [0,1]$$에 따라 결정된다. 그리고 유한한 step을 가정한다.

$$
V^{\pi, \gamma}(s) = E[r_{t+1} + ... + r_{t+T} \lvert s_t = s] \ \text{for all} \ s \in S
$$

#### 2. action value function $$Q^{\pi, \gamma}(s,a)$$

action value function은 다음과 같이 정의된다.

$$
Q^{\pi, \gamma}(s,a) = \Sigma_{s' \in S} P(s' | s, a)[R(s,a,s') + \gamma(s') V^{\pi, \gamma}(s)]
$$

그리고 모든 state $$s$$에 대해

$$
V^{\pi, \gamma}(s) = \Sigma_{a \in A}\pi(a|s)Q^{\pi, \gamma}(s,a)
$$

가 성립한다.

#### 3. objective function

다음 scalar objective function $$J(\cdot)$$을 극대화하는 policy를 선택하는 것이 강화학습의 목표가 된다.

$$
J_\gamma (u) = \Sigma_{s \in S} d^b(s) V^{\pi_{u, \gamma}}(s) \\

d^b(s) = lim_{t \rightarrow \infty} P(s_t = s | s_0,b)
$$

이때 policy $$\pi_u : A X S \rightarrow [0,1]$$은 미분 가능한 임의의 weight vector $$u \in \rm I\!R^{N_u}, \ N_u \in \rm I\!N$$를 가지고 있고, $$\pi_u (a \lvert s) > 0$$이 성립한다. 그리고 $$d^b(s)$$는 b에 의해 제약되는 state 분포를 의미하고, $$P(s_t = s \lvert s_0, b)$$는 $$s_0$$에서 시작하고 $$b$$를 실행했을 때 $$s_t = s$$가 성립할 확률을 뜻한다.

### Off-PAC

논문에서는 Off-poicy Actor-Critic을 줄여서 Off-PAC이라고 부르고 있다. Actor-Critic을 기본으로 하는 만큼 Policy weight 값을 업데이트하는 Actor 부분과 계산에 필요한 value function을 추정하는 Critic 부분 두 가지로 나누어져 있다.

전체적으로 Off-Policy 알고리즘의 유도과정은 아래 세 단계를 따른다.

#### 1. Critic: Policy Evaluation

Off-PAC의 critic의 업데이트는 Maei(2011)의 GTD(Gradient TD)방식을 사용한다고 한다. 자세한 내용은 현재 사용하고 있는 방식과는 크게 차이가 있는 것 같아 건너뛰기로 한다.

#### 2. Off-policy Policy-Gradient Theorem

다른 PG 방식과 마찬가지로 off-PAC 또한 objective의 기울기에 비례하여 업데이트가 이뤄진다.

$$
u_{t+1} - u_t \approx \alpha_{u,t} \nabla_u J_\gamma(u_t)
$$

이를 위해 위의 objective function의 기울기를 구해야 하는데 그 전개 과정은 다음과 같다.

$$
\eqalign{

\nabla_u J_\gamma (u) &= \nabla_u [\Sigma_{s \in S} d^b (s) \Sigma_{a \in A} \pi (a \lvert s) Q^{\pi, \gamma}(s,a)] \\

&= \Sigma_{s \in S} d^b (s) \Sigma_{a \in A} [ \nabla_u \pi (a \lvert s) Q^{\pi, \gamma}(s,a) + \pi (a \lvert s) \nabla_u Q^{\pi, \gamma}(s,a)]
}
$$

하지만 여기서 마지막 term $$\nabla_u Q^{\pi, \gamma}(s,a)$$는 off-policy로는 추정하는 것이 어렵다. 따라서 이를 생략할 필요가 있는데, 이를 위해 논문에서는 $$\nabla_u J_\gamma (u)$$의 근사로서 $$\nabla_u Q^{\pi, \gamma}(s,a)$$ 부분을 제외한 $$g(u) \in \rm I\!R^{N_u}$$를 사용한다.

$$
\nabla_u J_\gamma (u) \approx g(u) = \Sigma_{s \in S} d^b(s) \Sigma_{a \in A} \nabla_u \pi (a \lvert s) Q^{\pi, \gamma}(s,a)
$$

그리고 이러한 근사가 가능한 이유를 보이기 위해 아래 두 Theorem을 제시한다.

##### 1. Theorem 1: Policy Improvement

$$
\begin{multline}

\shoveleft \text{Given any policy parameter} \ u, \ \text{let} \ u'= u + \alpha g(u) \\

\shoveleft \text{Then, there exists an} \ \epsilon > 0 \ \text{such that, for all positive} \ \alpha < \epsilon, \ J_\gamma(u') \geqq J_\gamma(u) \\

\shoveleft \text{Further, if} \ \pi \ \text{has a tabular representation,(i.e. separate weights for each state),} \\
\shoveleft \text{then} \ V^{\pi_{u'}, \gamma}(s) \geqq V^{\pi_u, \gamma}(s) \ \text{for all} \ s \in S

\end{multline}
$$

policy parameter $$u$$를 갖는 objective function approximation $$g(u)$$가 있을 때, 그에 맞는 방향으로 u에 작은 숫자를 더한 결과를 $$u'$$라 하고 이를 실제 objective function $$J_\gamma(\cdot)$$에 넣은 결과가 기존보다 더 크면 policy improvement가 이뤄지고 있는 것으로 볼 수 있다는 것이다.

간단히 말해서 objective function의 gradient라고 가정하는 방향($$g(\cdot)$$)이 실제 objective function을 maximize 하는 방향과 동일하다는 것을 의미한다.

##### 2. Off-Policy Policy-Gradient Theorem

"Given $$U \subset \rm I\!R^{N_u}$$ a non-empty, compact set, let

$$
\eqalign{
& \tilde Z = \{ u \in U \lvert g(u) = 0 \} \\
& Z = \{ u \in U \lvert \nabla_u J_\gamma(u) = 0 \}
}
$$

where $$Z$$ is the true set of local maxima and $$\tilde Z$$ the set of local maxima obtained from using the approximate gradient, $$g(u)$$. If the value function can be represented by our function class, then, $$Z \subset \tilde Z$$. Moreover, if we use a tabular representation for $$\pi$$, then $$Z = \tilde Z$$"

제 2 정리에서는 tabular representation의 경우 $$Z = \tilde Z$$라고 하고 있다. 이는 증명 과정에서 $$Z = \tilde Z$$가 성립하기 위해서는 중복된 업데이트가 이뤄지지 않는, 즉 하나의 parameter에 대한 업데이트가 어떤 한 state에서 선택하는 action의 확률값만을 변화시키는 경우여야 한다는 것을 보이고 있기 때문이다. 결과적으로 두 개의 gradient $$\nabla_u J_\gamma(u)$$와 $$g(u)$$가 지역적으로 하나의 state에 대해서만 action의 확률을 바꿔야 한다는 것을 의미한다. 이를 그대로 적용하면 tabular가 아니더라도 한 번의 업데이트가 최소한의 작은 영역의 action-value function에만 영향을 미쳐야 한다.

운이 좋게도 최적화의 관점에서 다음이 성립한다.

$$
\text{for all} \ u \in \tilde Z \setminus Z, \ J_\gamma(u) < min_{u' \in Z} J_\gamma (u')
$$

즉, $$Z$$가 objective function $$J_\gamma$$의 관점에서 $$\tilde Z$$ 내의 모든 largest local maxima를 의미한다. 이 경우 random start와 같은 local optimization 방법들이 larger maxima로 수렴을 돕게 된다. objective function $$J_\gamma$$가 convex하지 않기 때문에 이러한 방법이 더욱 도움이 된다.

#### 3. Actor: Incremental Update Algorithm with Eligibility Traces
