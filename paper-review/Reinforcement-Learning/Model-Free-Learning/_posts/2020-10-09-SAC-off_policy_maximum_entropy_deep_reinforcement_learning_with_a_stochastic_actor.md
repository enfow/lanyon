---
layout: post
title: "Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
category_num: 12
keyword: '[SAC]'
---

# 논문 제목 : Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
- 2018
- [논문 링크](<https://arxiv.org/abs/1801.01290>)
- 2020.10.09

## Summary

- Actor-Critic 구조에 Maximum Entropy Framework를 적용한 SOTA 알고리즘 SAC를 제안한다.
- **Maximum Entropy Framework**란 Objective Function에 Policy의 Entropy Term을 추가하여 Exploration과 Robustness 측면에서 장점을 가지는 방법론이다.
- 기존의 Soft Q-Learning과 달리 Intractable한 Policy를 가우시안 중 하나가 되도록 **KLD**를 사용해 Projection하며, 업데이트 과정에서 **Reparameterization Trick**을 적용해 Gradient Descent가 가능하도록 만든다는 특징을 가진다.

## Problems of Actor-Critic

**Actor-Critic** 알고리즘의 장점 중 하나는 TRPO, PPO와 같은 On-Policy 알고리즘과 비교해 Off-Policy 방식을 따르므로 Sample Efficiency가 높다는 것이다. On-Policy 알고리즘은 매 Gradient Step 마다 새로운 Experience를 요구하는데 반해 Off-Policy 알고리즘은 현재 Policy가 아닌 과거 Policy에 의한 Experience도 학습에 사용할 수 있기 때문이다.

그러나 Actor-Critic 알고리즘 또한 Model-Free 알고리즘이 가지는 두 가지 문제, 수렴이 불안정적이고 여러 하이퍼파라미터에 대해 민감하게 반응한다는 것을 극복하는 데에 여전히 어려움을 겪고 있다. 특히 이러한 문제는 State와 Action Space가 Continuous하고 높은 차원으로 이뤄져 있을 때 보다 심각해지는 것으로 알려져 있다.

## Maximum Entropy Framework

**SAC(Soft Actor-Critic)**은 당연하게도 이러한 Actor-Critic의 문제를 해결하기 위해 제시된 알고리즘이다. 이를 위해 Ziebart의 논문 Maximum Entropy Inverse Reinforcement Learning(2008)을 비롯하여 강화학습에서 다양하게 적용되어 온 **Maximum Entropy Framework**를 도입한다.

Maximum Entropy Framework는 쉽게 말해 강화학습 모델을 학습하는 데 있어 Policy의 Entropy를 높이는 것을 고려하는 방법을 말한다. 강화학습에서 꽤나 오랫동안 사용되어온 방법론인 만큼 Maximum Entropy Framework의 장점 또한 많이 알려져 있으며, 특히 보다 효율적으로 탐색(Exploration)을 수행하고 모델이 보다 Robust 해진다고 알려져 있다. 그러나 [Soft Q-Learning](<https://arxiv.org/pdf/1702.08165.pdf>)을 비롯해 이를 적용한 많은 Off-Policy 알고리즘들은 Continuous Action Space 상에서 Approximate Inference를 수행하는 데 어려움을 겪어 이를 제대로 활용하지 못했다고 지적한다. 이러한 점에서 SAC는 Maximum Entropy Framework를 Actor-Critic에 적용하고 그 과정에서 안정적인 학습을 위해 제시하는 근사 알고리즘에 더 큰 의의를 가진다고 할 수 있다.

### Objective Function with Maximum Entropy

일반적인 강화학습의 목적함수는 다음과 같이 Reward 기대값의 총합으로 정의되며, 이를 극대화하는 방향으로 학습하게 된다.

$$
J(\pi) = \Sigma_t E_{(s_t, a_t) \backsim \rho_\pi} [r(s_t, a_t)]
$$

반면 Maximum Entropy Framework 에서는 Policy의 Entropy를 높이는 것에도 관심을 가진다. 여기서 $$\alpha$$는 Entropy Term이 얼마나 반영될지 결정하는 Temperature Parameter로, 이것이 0이 되면 위의 일반적인 강화학습 목적함수와 동일해진다.

$$
J(\pi) = \Sigma_t E_{(s_t, a_t) \backsim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal H (\pi( \cdot \lvert s_t) )]
$$

여기서 Entropy Term $$\mathcal H (\pi( \cdot \lvert s_t) )$$는 일반적인 Entropy의 정의에 따라 다음과 같이 구해진다.

$$
\mathcal H (\pi( a_t \lvert s_t) ) = - \log \pi( a_t \lvert s_t)
$$

이와 같이 목적함수에 Policy의 Entropy를 반영하도록 하면 다음 세 가지 장점을 가진다.

- 탐색을 보다 폭넓게 하는 것의 유인이 생긴다.
- Optimial Behavior과 유사한 여러 행동들을 찾을 수 있다.
- 경험적으로 탐색을 더 많이 하고 학습속도가 더 빠르다고 알려져 있다.

논문의 목표는 Actor-Critic 구조에 이러한 목적 함수를 적용하여 학습을 진행하는 것이다. 이를 위해 먼저 Actor-Critic 구조에 이를 적용했을 때 Optimal Policy로 수렴하는지 여부를 수학적으로 확인해 볼 필요가 있다.

## Soft Policy Iteration

이를 위해 논문에서는 Soft Policy Iteration을 도출하는 것에서부터 출발한다. **Soft Policy Iteration**이란 Maximum Entropy Policy를 찾기 위해 Policy Iteration, 즉 Policy Evaluation과 Policy Improvement를 반복하는 방법을 말한다.

### Soft Policy Evaluation

Policy Evauluation은 말 그대로 현재 Policy에 대한 Value Function을 최대한 정확히 구해 얼마나 좋은 Policy인지 평가하는 과정이다. 일반적인 Action Value Function $$Q$$는 Policy $$\pi$$에 따라 행동하는 Agent가 어떤 State에서 어떤 Action을 했을 때 기대되는 누적 Reward로 정의된다. 표현상의 편의를 위해 논문과 같이 Bellman Operator를 사용했다.

$$
\mathcal T^\pi Q(s_t, a_t) = r(s_t, a_t) + \gamma E_{s_{t+1} \backsim p, a_{t+1} \backsim \pi} [Q(s_{t+1}, a_{t+1})]
$$

하지만 Soft Policy Evaluation에서는 Maximum Entropy Policy를 고려해야 하므로 다음과 같이 정의된다.

$$
\mathcal T^\pi Q(s_t, a_t) = r(s_t, a_t) + \gamma E_{s_{t+1} \backsim p} [V(s_{t_1})] \\
V(s_t) = E_{a_t \backsim \pi} [Q(s_t, a_t) - \log \pi (a_t \lvert s_t)]
$$

$$V(s_t)$$를 계산하는 데 있어 Entropy Term $$- \log \pi (a_t \lvert s_t)$$이 추가되었으며, 이를 **Soft State Value Function**이라고 부른다. Soft Policy Evaluation이 가능하다는 것을 보이기 위해서는 Policy $$\pi$$가 이 Soft State Value Function $$V(s_t)$$에 수렴한다는 것을 보여야 한다. 논문에서는 $$\lvert \mathcal A \rvert < \infty$$를 가정하고 Reward를

$$
r(s_t, a_t) \triangleq r(s_t, a_t) + \gamma E_{s_{t+1} \backsim p, a_{t+1} \backsim \pi} [Q(s_{t+1}, a_{t+1})]
$$

라고 한다면 일반적인 Policy Evaluation과 동일해진다는 점에서 수렴성을 보이고 있다. 논문에 나와있는 Lemma는 다음과 같다.

$$
\eqalign{
&\text{[Lemma 1] Soft Policy Evaluation} \\
&\text{ Consider the soft Bellman Backup Operator} \mathcal T^{\pi} \text{ and a mapping } Q^0: \mathcal S \times \mathcal A \rightarrow R \\
&\text{with } \lvert A \rvert < \infty, \text{ and define } Q^{k+1} = \mathcal T^\pi Q^k. \text{ Then the sequence } Q^k \text{ will converge }\\
&\text{to the soft Q-value of } \pi \text{ as } k \rightarrow \infty
}
$$

### Soft Policy Improvement

Soft Policy Improvement에서는 현재의 Policy를 Exponential을 취한 $$\exp(Q^{\pi_{old}}(s_t, \cdot))$$에 따라 업데이트하게 된다. 그런데 문제가 하나 있다면 $$\exp(Q^{\pi_{old}}(s_t, \cdot))$$의 분포가 Intractable 하다는 것이다. 이를 해결하기 위해 Policy를 가우시안과 같이 Tractable한 함수들의 집합 $$\Pi$$의 원소로 제한하고 업데이트 된 새로운 $$\pi^{new}$$는 그 중에서 $$\exp(Q^{\pi_{old}}(s_t, \cdot))$$와 가까운 것을 고르도록(Projection) 하고 있다. 논문에서는 분포 간의 차이를 측정하는 가장 대표적인 방법 중 하나인 Kullback-Leibler Divergence를 줄이는 방향으로 업데이트 식을 도출한다.

$$
\pi_{new} = \arg \min_{\pi' \in \Pi} \text{KLD} ( \pi' (\cdot \lvert s_t) \| { \exp (Q^{\pi_{old}}(s_t, \cdot)) \over Z^{\pi_{old}} (s_t)} )
$$

이때 Normalizer로 도입된 Partition function $$Z(s_t)$$는 일반적으로 Intractable 하지만 Gradient에 영향을 주지 않으므로 업데이트 과정에서는 무시된다. 논문에서는 이를 Lemma2로 정의하고 있다.

$$
\eqalign{
&\text{[Lemma 2] Soft Policy Improvement} \\
&\text{ Let } \pi_{old} \in \Pi \text{ and let } \pi_{new} \text{ be the optimizer of the minimization problem of }\\
& \qquad \qquad \pi_{new} = \arg \min_{\pi' \in \Pi} \text{KLD} ( \pi' (\cdot \lvert s_t) \| { \exp (Q^{\pi_{old}}(s_t, \cdot)) \over Z^{\pi_{old}} (s_t)} )\\
& \text{Then } Q^{\pi_{new}}(s_t, a_t) \geq Q^{\pi_{old}}(s_t, a_t) \text{ for all } (s_t, a_t) \in \mathcal S \times \mathcal A \text{ with } \lvert \mathcal A \rvert < \infty
}
$$

증명은 다음과 같은 KLD의 특성에서 출발한다.

$$
\begin{multline}

\shoveleft \text{KLD} ( \pi_{new} (\cdot \lvert s_t) \| { \exp (Q^{\pi_{old}}(s_t, \cdot)) \over Z^{\pi_{old} (s_t)}} ) \leq \text{KLD} ( \pi_{old} (\cdot \lvert s_t) \| { \exp (Q^{\pi_{old}}(s_t, \cdot)) \over Z^{\pi_{old} (s_t)}} )\\
\shoveleft \Rightarrow \ E_{a_t \backsim \pi_{new}} [\log \pi_{new} (a_t \lvert s_t) - Q^{\pi_{old}} + \log Z^{\pi_{old}}(s_t) ] \leq E_{a_t \backsim \pi_{old}} [\log \pi_{old} (a_t \lvert s_t) - Q^{\pi_{old}} + \log Z^{\pi_{old}}(s_t) ] \\
\shoveleft \Rightarrow \ E_{a_t \backsim \pi_{new}} [\log \pi_{new} (a_t \lvert s_t) - Q^{\pi_{old}} ] \leq E_{a_t \backsim \pi_{old}} [\log \pi_{old} (a_t \lvert s_t) - Q^{\pi_{old}}] \qquad \because \ Z^\pi \text{ only depend on } s\\
\shoveleft \Rightarrow \ E_{a_t \backsim \pi_{new}} [\log \pi_{new} (a_t \lvert s_t) - Q^{\pi_{old}} ] \leq - V^{\pi_{old}} (s_t) \\
\shoveleft \Rightarrow \ E_{a_t \backsim \pi_{new}} [Q^{\pi_{old}} - \log \pi_{new} (a_t \lvert s_t) ] \geq V^{\pi_{old}} (s_t)\\
\end{multline}
$$

이 식을 Bellman Equation $$Q^{\pi_{old}}(s_t, a_t) = r(s_t,a_t) + \gamma E_{s_{t+1} \backsim p} [V^{\pi_{old}}(s_{t+1})] $$에 적용하면 $$Q_{\pi_{old}} \leq Q_{\pi_{new}}$$가 성립한다는 것을 알 수 있다.

## Soft Actor-Critic

Soft Policy Iteration을 Actor-Critic에 적용하기 위해 State Value Function과 Action Value Function, Policy를 Neural Net으로 모사하고 각각을 $$V_\psi, Q_\theta, \pi_\phi$$로 표기한다. 또한 Value Iteration, 즉 Policy Evaluation과 Policy Improvement 각각을 수렴할 때까지 계속해서 반복 수행하지 않고 한 번씩만 하도록 하고 있다.

### Objective Function for $$V_\psi$$

State Value Function $$V_\psi$$의 목표는 위에서 정의한 Soft Value Function을 근사하는 것이다.

$$
V(s_t) = E_{a_t \backsim \pi} [Q(s_t, a_t) - \log \pi (a_t \lvert s_t)]
$$

따라서 $$V_\psi$$와 위 식 간의 차이를 줄이는 방향으로 목적 함수가 정의된다.

$$
J_V(\psi) = E_{s_t \backsim D} [{1 \over 2} (V_\psi (s_t) - E_{a_t \backsim \pi_\phi} [Q_{\theta} (s_t, a_t) - \log \pi_\phi (a_t \lvert s_t)] )^2 ]
$$

여기서 $$s_t$$를 샘플링하는 분포 $$D$$는 Replay Buffer이다. 이는 Continuous State Space에서는 가능한 State가 무한하다는 것을 고려한 것이다.

### Objective Function for $$Q_\theta$$

Action Value Function의 Target은 다음과 같다.

$$
\hat Q(s_t, a_t) = r(s_t, a_t) + \gamma E_{s_{t+1} \backsim p} [V_{\bar \psi} (s_{t+1}) ]
$$

여기서 $$V_{\bar \psi} (s_{t+1})$$는 $$V_{\psi} (s_{t+1})$$의 지수이동평균으로, 안정적인 학습을 위한 것이라고 할 수 있다.

$$
J_Q(\theta) = E_{(s_t,a_t) \backsim D} [{1 \over 2} (Q_\theta(s_t, a_t) - \hat Q (s_t, a_t))^2 ]
$$

### Objective Function for $$\pi_\phi$$

Policy $$\pi_\phi$$는 위에서 보았던 것처럼 KLD를 줄이는 방향으로 구해진다.

$$
J_\pi (\phi) = E_{s_t \backsim D} [KLD ( \pi_{\phi} (\cdot \lvert s_t) \| { {\exp (Q_\theta {(s_t, \cdot))}} \over { Z_{\theta} (s_t)} }) ]
$$

이때 $$\pi_\phi$$와 $$Q_\theta$$는 Neural Net이므로 Gradient Descent로 업데이트 되어야 한다. 따라서 $$\pi_{\phi} (\cdot \lvert s_t)$$ 또한 미분 가능하게 만들 필요가 있다. $$\pi_{\phi} (\cdot \lvert s_t)$$는 가우시안이므로 VAE에서 사용되는 **Reparameterization Trick**을 사용하면 쉽게 미분 가능한 형태로 표현할 수 있다.

$$
a_t = f_\phi (\epsilon_t; s_t)
$$

이를 적용한 목적 함수는 다음과 같다.

$$
J_\pi(\phi) = E_{s_t \backsim D, \epsilon_t \backsim N} [\log \pi_\phi (f_\phi(\epsilon_t; s_t) \lvert s_t ) - Q_\theta(s_t, f_\phi (\epsilon_t; s_t))]
$$

### Algorithm

Soft Actor-Critic의 알고리즘은 다음과 같이 환경으로부터 데이터를 수집하는 부분과 그것을 바탕으로 업데이트를 하는 부분 두 가지로 이뤄져 있다.

<img src="{{site.image_url}}/paper-review/sac_algorithm.png" style="width:30em; display: block; margin: 0px auto;">
