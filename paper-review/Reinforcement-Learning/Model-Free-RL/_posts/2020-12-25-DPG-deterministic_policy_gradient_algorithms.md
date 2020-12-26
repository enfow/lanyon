---
layout: post
title: "Deterministic Policy Gradient Algorithms"
category_num: 10
keyword: '[DPG]'
---

# 논문 제목 : Deterministic Policy Gradient Algorithms

- David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller
- 2018
- [논문 링크](<http://proceedings.mlr.press/v32/silver14.pdf?CFID=6293331&CFTOKEN=eaaee2b6cc8c9889-7610350E-DCAB-7633-E69F572DC210F301>)
- [보충 자료](<http://proceedings.mlr.press/v32/silver14-supp.pdf>)
- 2020.12.25

## Summary

- Stochastic Policy $$\pi(a \lvert s)$$가 아닌 **Deterministic Policy $$\mu(s)$$에서도 Policy Gradient Theorem이 성립**함을 보이고 있다.
- Deterministic Policy란 결국 Stochastic Policy에서 Variance가 0인 특수한 경우라는 것을 증명하고, 이를 통해 Stochastic Policy Gradient가 적용되는 기존 방법론(Actor-Critic)에서도 Deterministic Policy를 사용할 수 있다는 것을 보이고 있다.
- Actor-Critic에서 Critic을 Function Approximator $$Q^w(s,a)$$로 하여 Performance Gradient를 구할 때, True q-Function이 아니어서 발생하는 Bias를 제거하는 조건을 제시하고 있다.

## Policy Gradient Theorem

**Policy Gradient**란 말 그대로 Policy $$\pi(a \lvert s)$$를 Parameterize하고, Performance를 극대화하는 방향으로 이를 업데이트하는 강화학습 방법론을 말한다. 이때 Performance라는 것은 다음과 같이 현재의 Policy $$\pi_\theta$$를 따를 때 얻을 것으로 기대되는 Reward의 총합으로 정의된다.

$$
\eqalign{
J(\pi_\theta) &= \int_S \rho^\pi(s) \int_A \pi_\theta (s,a) r(s,a) dads\\
&= E_{s\backsim \rho^\pi, a \backsim \pi_\theta} [r(s, a)]
}
$$

이를 극대화하는 방향이라고 할 수 있는 Gradient는 다음과 같이 구해진다. 이를 **Policy Gradient Theorem**이라고 한다.

$$
\eqalign{
\nabla J(\pi_\theta) &= \int_S \rho^\pi(s) \int_A \nabla_\theta \pi_\theta (s,a) Q^\pi(s,a) dads\\
&= E_{s\backsim \rho^\pi, a \backsim \pi_\theta} [\nabla_\theta \log \pi_\theta(a \lvert s) Q^\pi(s,a)]
}
$$

Policy Gradient Theorem은 Performance Gradient를 단순한 기대값 형태로 바꾸어주어 적은 연산량으로 쉽게 추정할 수 있도록 해주었다는 점에서 실용적인 측면에서도 큰 의의를 가지고 있다. 위의 식에 따르면 Performance Gradient를 구할 때 State Distribution $$\rho^\pi$$가 고려되기는 하나, 어떤 State에서의 $$\nabla_\theta \log \pi_\theta(a \lvert s) Q^\pi(s,a)$$가 얼마나 중요한지 결정하는 것일 뿐 State Distribution의 Gradient 자체는 필요하지 않다.

### Actor-Critic Architecture

Actor-Crtic 구조는 Policy Gradient Theorem을 바탕으로 하는 알고리즘 중 하나로서, 이름으로 유추할 수 있듯이 Action을 결정하는 Actor $$\pi_\theta(s)$$와 선택한 Action이 얼마나 좋은지 평가하는 Critic $$Q^w(s,a)$$으로 이뤄져 있다. 이러한 점에서 Actor-Critic 구조는 Policy Gradient Theorem에서 True Q Function $$Q^\pi$$을 parameter $$w$$를 가지는 $$Q^w$$로 근사하여 대체하는 방법론이라고 할 수 있다.

$$
\eqalign{
&E_{s\backsim \rho^\pi, a \backsim \pi_\theta} [\nabla_\theta \log \pi_\theta(a \lvert s) Q^\pi(s,a)]\\
& \Rightarrow E_{s\backsim \rho^\pi, a \backsim \pi_\theta} [\nabla_\theta \log \pi_\theta(a \lvert s) Q^w(s,a)]
}
$$

True Q Function $$Q^\pi$$를 대신하여 근사함수 $$Q^w$$를 사용하여 학습을 진행하게 되면 Bias가 발생하게 되고 이로 인해 안정적인 학습이 방해받을 수 있다. 이와 관련하여 $$Q^w(s,a)$$가 다음 두 조건을 만족하는 경우 Bias는 존재하지 않는다는 것이 [Sutton 등](<http://incompleteideas.net/papers/SMSM-NIPS99-submitted.pdf>)에 의해 증명되어 있다.

$$
\eqalign{
& 1. Q^w(s,a) = \nabla_\theta \log \pi_\theta (a \lvert s)^\text{T} w \\
& 2. w \text{ is chosen to minimize } \epsilon^2(w) = E_{s \backsim \rho^\pi, a \backsim \pi_\theta} [(Q^w(s,a) - Q^\pi(s,a))^2]
}
$$

첫 번째 조건은 Compatible Function Apprximator $$Q^w(s, a)$$는 Policy의 Gradient $$\nabla_\theta \log \pi_\theta(a \lvert s)$$의 feature에 대해 선형의 관계를 가진다는 것으로, 두 번째 조건은 $$Q^w(s, a)$$로 $$Q^\pi(s, a)$$를 예측하는 선형회귀 문제로 풀어야 한다는 것으로 이해할 수 있다. Actor-Critic 구조에서 자주 사용하는 TD method를 통해 Critic을 업데이트하면 두 번째 조건을 만족하게 된다.

결과적으로 두 가지 조건을 모두 만족하게 되면 REINFORCE 알고리즘과 같이 Critic을 사용하지 않는 알고리즘들과 동일해진다고 한다.

### Off-Policy Actor-Critic Architecture

Policy Gradient Thoerem에서 $$\rho^\pi(s)$$는 **On-Policy State Distribution**으로, 현재의 Policy $$\pi$$를 따를 때 어떤 State $$s$$를 방문할 확률을 의미한다. 이와 달리 State Distribution $$\rho^\beta(s)$$처럼 현재 Policy $$\pi$$가 아닌 다른 Policy $$\beta$$를 사용할 수도 있는데, 이를 **Off-Policy State Distribution** 이라고 한다. 그리고 두 Policy를 각각 $$\beta$$를 **Behavior Policy**, $$\pi$$를 **Target Policy**라고 한다.

$$
\eqalign{
J_\beta (\pi_\theta) &= \int_S \rho^\beta (s) V^\pi(s) ds\\
&= \int_S \int_A \rho^\beta (s) \pi_\theta(a \lvert s) Q^\pi (s, a) da ds
}
$$

이때 Performance Gradient는 다음과 같이 구할 수 있다.

$$
\eqalign{
\nabla J_\beta (\pi_\theta) &= \nabla_\theta \int_S \int_A \rho^\beta (s) \pi (s \lvert a) Q^\pi (s, a) da ds\\
&= \int_S \int_A \rho^\beta(s) [\nabla_\theta \pi(a \lvert s) Q^\pi(s,a) + \pi(a \lvert s)\nabla_\theta Q^\pi(s,a)] da ds \\
&\approx \int_S \int_A \rho^\beta(s) [\nabla_\theta \pi(a \lvert s) Q^\pi(s,a)] da ds \\
&= E_{s \backsim \rho^\beta, a \backsim \beta} [ {\pi_\theta (a \lvert s) \over {\beta_\theta (a \lvert s)}} \nabla_\theta \log \pi_\theta(a \lvert s) Q^\pi(s, a)]
}
$$

위 전개 식의 가장 큰 특징은 세 번째 줄에서 $$\pi(a \lvert s)\nabla_\theta Q^\pi(s,a)$$ Term이 사라졌다는 데에 있다. Action Value Function에 대한 Gradient를 구하는 것이 까다롭기 때문에 제거한 것인데, [Degris 등](<https://arxiv.org/pdf/1205.4839.pdf>)에 따르면 이 때에도 Local Optima로의 수렴성이 보장된다고 한다. 마지막 줄의 $${\pi_\theta (a \lvert s) \over {\beta_\theta (a \lvert s)}}$$는 $$\beta$$를 따를 때의 분포와 $$\pi$$를 따를 때의 분포가 다르기 때문에 이를 맞춰주는 Importance Sampling Term이다.

## Deterministic Policy Gradient

지금까지 살펴본 Policy Gradient Theorem과 그 변형들은 모두 **Stochastic Policy $$\pi(a \lvert s)$$**를 가정하고 있다. 같은 $$s$$에 대해서도 다양한 $$a$$가 확률적으로 결정될 수 있다는 점에서 Stochastic이라는 표현이 붙었다. 반면 논문에서 다루고 있는 주제인 **Deterministic Policy Gradient**는 **Deterministic Policy $$\mu(s)$$**, 즉 하나의 State $$s$$는 항상 하나의 Action $$a$$로 매핑되는 함수를 사용할 때의 Policy Gradient를 말한다.

이와 같이 Deterministic Policy를 사용하게 되면 당장 두 가지에 대한 의문이 발생하게 된다. 첫 번째는 수렴성에 대한 증명, 즉 Gradient Theorem이 이 경우에도 만족하는지 확인해야 할 필요가 있다는 것이고, 두 번째는 Exploration을 수행하는 방법에 관한 것이다. 논문에서는 이에 대한 답변으로 (1) Deterministic Policy란 Stochastic Policy의 특수한 형태($$\sigma = 0$$)일 뿐이고, (2) 앞서 Stochastic한 경우에서 수렴성 증명이 완료되어 있는 Behavior Policy를 사용하는 것으로 Exploration을 도입할 수 있다고 말한다.

### Deterministic Policy Gradient Theorem

Stochastic Policy를 고려할 때와 가장 큰 자이점은 Deterministic Policy에서는 더 이상 다양한 Action에 대한 경우의 수를 고려할 필요가 없다는 것이다. 따라서 Performance Objective 또한 다음과 같이 다시 쓸 수 있다

$$
\eqalign{
\text{Stochastic:   }& \eqalign{
J(\pi_\theta) &= \int_S \rho^\pi(s) \int_A \pi_\theta (s,a) r(s,a) dads\\
&= E_{s\backsim \rho^\pi, a \backsim \pi_\theta} [r(s, a)]
}
\\
\\
\text{Deterministic:   }& \eqalign{
J(\pi_\theta) &= \int_S \rho^\pi(s) r(s,\mu_\theta(s)) ds\\
&= E_{s\backsim \rho^\pi,} [r(s, \mu_\theta(s))]
}
}
$$

논문에서 제시하는 **Deterministic Policy Gradient Theorem**은 다음과 같다.

$$
\begin{multline}
\shoveleft{ \text{Suppose the the MDP satisfies A.1 conditions below,}}\\
\end{multline}
$$

$$
\begin{multline}
\shoveleft {\text{Regularity Confitions }}\\
\shoveleft {
    \quad \text{A.1: } p(s' \lvert s, a), \nabla_a p(s' \lvert s, a), \mu_\theta(s), \nabla_\theta \mu_\theta (s), r(s,a), \nabla_a r(s,a), p_1(s) \text{ are continuous }
}\\
\shoveleft{
    \qquad \text{in all parameters and variables } s, a s' and x.
}\\
\shoveleft{
    \quad \text{A.2: there existis a } b \text{ and } L \text{ such that } \sup_s p_1(s) < b, \sup_{a,s,s'} p(s' \lvert s, a) < b, \sup_{a,s} r(s, a) < b,
}\\
\shoveleft{
    \qquad \sup_{a,s,s'} | \nabla_a p(s' \lvert s, a) | < L \text{ and } \sup_{a,s} | \nabla_a r(s, a) | < L
}\\
\end{multline}
$$

$$
\begin{multline}
\shoveleft{ \text{then, } }\\

\shoveleft{ \qquad \nabla_\theta J(\mu_\theta) = \int_S \rho^\mu (s) \nabla_\theta \mu_\theta (s) \nabla_a Q^\mu (s, a) \lvert_{a = \mu_\theta(s)} }\\
\end{multline}
$$

이때 일정한 조건을 만족하는 경우에 한해 Variance $$\sigma$$가 0으로 수렴하면($$\sigma \rightarrow 0$$) Stochastic Policy Gradient가 Deterministic Policy Gradient Theorem으로 수렴한다는 것 또한 증명이 가능하다고 한다.

$$
\begin{multline}
\shoveleft{
    \text{Consider a stochastic Policy } \pi_{\mu_\theta, \sigma} \text{ such that } \pi_{\mu_\theta, \sigma} (a \lvert s) = v_\sigma(\mu_\theta (s), a)
}\\
\shoveleft{
    \text{where } \sigma \text{ is a parameter controlling the variance and } v_\sigma \text{ satisfy condition}
}\\
\shoveleft{
    \text{B.1(see supplementary) and the MDP satisfies conditions A.1, A.2.}
}\\
\shoveleft{
    \text{Then, } \lim_{\sigma \rightarrow 0} \nabla_\theta J(\pi_{\mu_\theta, \sigma}) = \nabla_\theta J(\mu_\theta)
}\\
\end{multline}
$$

이에 따라 Stochasitic Policy Gradient Theorem에서 파생되어 나온 다양한 Application(Actor-Critic, Natural Gradient, Compatible Function Approximation 등)에도 Deterministic Policy를 적용할 수 있다는 것을 알 수 있다.

### On-Policy Deterministic Actor-Critic

SARSA 업데이트를 사용하는 On-Policy Deterministic Actor-Critic의 업데이트 식을 다음과 같이 정리할 수 있다.

$$
\eqalign{
    \delta_t &= r_t + \gamma Q^w (s_{t+1}, a_{t+1}) - Q^w(s_t, a_t)\\
    w_{t+1} &= w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)\\
    \theta_{t+1} &= \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_a Q^w (s_t, a_t) \lvert_{a = \mu_\theta(s)}
}
$$

### Off-Policy Deterministic Actor-Critic

Off-Policy에서는 Stochastic Actor-Critic을 참고하여 다음과 같이 Performance Gradient를 정의할 수 있다.

$$
\eqalign{
\nabla_\theta J_\beta(\mu_\theta) &\approx \int_S \rho^\beta(s) [\nabla_\theta \mu_\theta(s) Q^\mu(s,a)] ds \\
&= E_{s \backsim \rho^\beta} [\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a) \lvert_{a = \mu_\theta(s)} ]
}
$$

Stochastic Actor-Critic 식과 비교해 볼 때 가장 큰 차이점은 Importance Sampling Term이 사라졌다는 것이다. 이는 Deterministic Policy에서는 동일한 State에 대해 더 이상 다양한 Action을 고려할 필요가 없기 때문이다.

위 식에 근거하는 **Off-Policy Deterministic Actor-Critic(OPDAC)**의 업데이트 식은 다음과 같다. 여기서는 Actor를 업데이트할 때 SARSA가 아닌 Q-learning을 사용하고 있다.

$$
\eqalign{
    \delta_t &= r_t + \gamma Q^w (s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)\\
    w_{t+1} &= w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)\\
    \theta_{t+1} &= \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_a Q^w (s_t, a_t) \lvert_{a = \mu_\theta(s)}
}
$$

### Compatible Function Approximation

위의 Deterministic Actor-Critic에서도 Stochastic에서와 마찬가지로 근사 함수 $$Q^w(s,a)$$의 Bias 문제가 발생할 수 있다. 논문에 따르면 아래와 같은 조건을 만족하면 Deterministic Actor-Critic에서도 Performacne Gradient에 영향을 주지 않으면서 $$\nabla_a Q^\mu(s,a)$$를 대체할 수 있는 Critic $$Q^w$$를 찾을 수 있다고 한다.

$$
\begin{multline}
\shoveleft{
    \text{A function approximator } Q^w(s, a) \text{ is compatible with a deterministic policy } \mu_\theta(s),
}\\
\shoveleft{
    \nabla_\theta J_\beta(\theta) = E[\nabla_\theta \mu_\theta(s) \nabla_a Q^w(s,a) \lvert_{a = \mu_\theta (s)}], \text{ when satisfy conditions below.}
}\\
\shoveleft{ \quad
    \text{1. } \nabla_a Q^w (s,a) \lvert_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta (s)^\text{T} w
}\\
\shoveleft{ \quad
    \text{2. } w \text{ minimises the MSE} (\theta, w) = E[\epsilon(s; \theta, w)^\text{T} \epsilon(s; \theta, w)] \text{ where }
}\\
\shoveleft{
    \qquad \epsilon(s; \theta, w) = \nabla_a Q^w(s, a) \lvert_{a = \mu_\theta(s)} - \nabla_a Q^\mu(s, a) \lvert_{a = \mu_\theta(s)}
}
\end{multline}
$$

증명은 $$\text{MSE} (\theta, w)$$가 0이 될 때 최소가 된다는 점에서 시작한다.

$$
\eqalign{
    \nabla_w \text{MSE} (\theta, w) &= 0\\
    E[\epsilon(s; \theta, w)^\text{T} \epsilon(s; \theta, w)] & = 0 \\
    E[\nabla_\theta \mu_\theta (s) \epsilon(s; \theta, w)] & = 0 \\
    E[\nabla_\theta \mu_\theta (s) (\nabla_a Q^w(s, a) \lvert_{a = \mu_\theta(s)} - \nabla_a Q^\mu(s, a) \lvert_{a = \mu_\theta(s)})] & = 0 \\
    E[\nabla_\theta \mu_\theta (s) \nabla_a Q^w(s, a) \lvert_{a = \mu_\theta(s)}] & = E[\nabla_\theta \mu_\theta (s) \nabla_a Q^\mu(s, a) \lvert_{a = \mu_\theta(s)}\\
    \therefore E[\nabla_\theta \mu_\theta (s) \nabla_a Q^w(s, a) \lvert_{a = \mu_\theta(s)}] & = \nabla_\theta J_\theta (\mu_\theta) \text{ or } \nabla_\theta J(\mu_\theta)\\
}
$$

이때 첫 번째 조건은 다음과 같은 형태로 $$Q^w$$를 정의하면 만족한다.

$$
Q^w(s,a) = (a - \mu_\theta(s))^{\text{T}} \nabla_\theta \mu_\theta(s)^{\text{T}} w + V^v(s)
$$

여기서 $$V_v(s)$$는 Action에 독립적인 함수로서 State-Value를 근사하게 된다. 따라서 위 식은 $$Q$$와 $$V$$에 관한 식으로 볼 수 있으므로 우변의 첫 번째 Term은 Advantage $$A$$ Term이 된다.

$$
\eqalign{
Q^w(s,a) - V^v(s) &= (a - \mu_\theta(s))^{\text{T}} \nabla_\theta \mu_\theta(s)^{\text{T}} w\\
A^w(s, a) &= \phi(s,a)^\text{T}w \\
\text{ where } &\phi(s,a) = \nabla_\theta \mu_\theta(s) (a-\mu_\theta(s))
}
$$

두 번째 조건에서는 $$\nabla_a Q^\mu(s,a)$$를 구하는 방법이 문제 된다. $$w$$를 SARSA 또는 Q-learning과 같은 기본적인 Policy Evaluation 방식을 통해 업데이트한다는 것만으로는 완벽하게 만족할 수 없기 때문이다. 이에 대해서는 어느 정도의 오차를 감수하고 $$Q^w(s,a) \approx Q^\mu(s,a)$$를 찾으면 $$\nabla_a Q^w(s,a) \lvert_{a = \mu_\theta(s)} \approx \nabla_a Q^\mu(s,a) \lvert_{a = \mu_\theta(s)}$$도 어느 정도 만족할 것이라는 선에서 정리하고 있다. 한 마디로 $$Q^w(s,a)$$의 $$w$$를 업데이트할 때 $$Q^\mu(s,a)$$에 근사하도록 업데이트하면 그것의 Gradient도 비슷해지는 것으로 보겠다는 것이다.
