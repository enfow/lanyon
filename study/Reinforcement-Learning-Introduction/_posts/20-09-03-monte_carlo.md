---
layout: post
title: Monte Carlo Methods
category_num: 5
---

# Monte Carlo Methods

- Sutton의 2011년 책 Reinforcement Learning: An Introduction 2nd edition을 참고해 작성했습니다.  
- update at : 2020.09.03

## 0. Introduction

**Monte Carlo Method**는 대표적인 강화학습 학습 방법론 중 하나로, 강화학습의 문제를 Sample Returns의 평균으로 해결하는 방법이다. **Return**이라는 표현에서도 알 수 있듯이 Value를 추정하거나 Policy를 업데이트 하기 위해서는 **Episode**가 끝나야 하며, 이러한 점에서 Monte Carlo Method를 step-by-step이 아닌 **episode-by-episode**한 특성을 가진다고 표현한다.

강화학습의 문제를 각 State-Action에 대한 Return들의 평균으로 해결한다는 점에서 Multi-Armed Bendit 문제와 유사하다고 할 수 있다. 하지만  Monte Carlo Method를 비롯하여 앞으로 보게 될 강화학습 알고리즘들은 State의 개수만큼 많은 수의 Multi-Armed Bendit 문제가 존재하는 환경에서 최적의 Policy를 찾는 것에 가깝다. 보다 구체적으로 단일의 State를 상정하는 Multi-Armed Bendit과 달리 Monte-Carlo Method는 기본적으로 Multi-State 환경을 가정하고 있고, 이전 State에서 결정한 Action에 현재 State가 영향을 받는다는 점에서 해결하고자 하는 문제의 상황 자체가 다르다.

## 1. Monte Carlo Prediction

어떤 Policy $$\pi$$에 대해 Value Function $$v_\pi(s)$$ 또는 $$q_\pi(s,a)$$를 추정한다는 것은 $$\pi$$를 따라 episode를 진행할 때 $$s$$에서 시작하는 경우 혹은 $$s$$에서 $$a$$를 선택하는 것에서 시작하는 경우 받을 것으로 기대되는 Return $$G$$를 추정하겠다는 것을 의미한다. Monte Carlo Method는 앞서 언급한대로 이를 매우 직관적으로 받아들여 현재 State $$s$$에서 Episode를 여러 번 진행하여 구한 Return들의 평균으로 Value를 예측하려는 접근 방법이다. 당연하게도 보다 정확한 예측을 위해서는 샘플, 즉 에피소드를 많이 확보해야 한다.

Episode를 진행하다보면 경우에 따라 동일한 State를 여러 번 거치게 될 수도 있다. 이와 관련하여 Return을 산정하는 방식이 다음과 같이 두 가지로 나누어진다.

- First Visit MC: 어떤 State를 처음 방문했을 때 만을 기준으로 Return을 결정한다.
- Every Visit MC: 어떤 State를 방문한 모든 경우의 Return을 평균한 것으로 State의 Return을 결정한다.

두 가지 방법 중 무엇으로 하든 무한 번 Episode를 뽑게 되면 $$v_\pi(s)$$로 수렴하게 된다고 한다.

### Monte Carlo for Action Value

Monte Carlo Method와 관련하여 한 가지 짚고 넘어가야 할 점이 있다면 Model의 정보, 즉 State Transition 등과 같은 환경에 대한 정보가 없다면 State Value Function $$v(s)$$만으로는 어떤 Action이 좋은지 정확하게 알 수 없다는 점이다. 어느 State가 좋은지 명확히 알고 있다 하더라도 어떤 Action을 취해야 해당 State로 갈 수 있는 지 알 수 없기 때문이다. 따라서 Model-Free의 문제를 해결할 때에는 State Value Function보다 Action Value Function $$q(s,a)$$가 더 유용하다.

하지만 $$q(s,a)$$를 추정하게 되면 경우의 수가 Action Space의 크기인 $$\lvert A \rvert$$배 만큼 증가하게 된다. 또한 단순히 갯수의 문제를 넘어 Action을 선택하는 방법도 달라져야 하는데, 어떤 State에서 현재 Policy가 결정하는 Action 외에 선택 가능한 다른 Action들도 경험하고 그에 대한 Value도 정확히 계산해야 하기 때문이다.

따라서 다양한 State-Action을 경헙해보는 것이 중요한데 Monte Carlo Method에서 이를 위한 해결 방법으로는 다음 두 가지가 대표적이라고 한다.

- 시작 시점에서의 State-Action을 임의로 선택하는 것(**Exploring Start**)
- 확률적으로 Policy가 Action을 결정하도록 하는 것(**Stochastic Policy**)

시작에 randomness를 부여하는가, 과정 상에서 randomness를 부여하는가의 차이라고 할 수 있다. 이때 중요한 것은 모든 State-Action 쌍의 선택 확률이 0이상이어서 어떤 쌍도 선택이 가능해야 한다는 점이다.

## 2. Monte Carlo Control

다양한 $$q(s,a)$$를 경험하는 것과 관련해 Exploring Start와 Stochastic Policy 두 가지 방법이 있다고 했었다. **Monte Carlo ES**는 이 중  **Exploring Start**를 적용하여 Policy Iteration을 수행하는 방법이다.

### 2.1. Monte Carlo ES

Monte Carlo Method는 Value Function을 추정하는 방법이라고 할 수 있는데, 이를 활용하여 Policy Evaluation을 진행하고 업데이트 된 Value Function에 따라 Policy를 업데이트하는 Policy Improvement를 진행하게 되면(Policy Iteration) 환경에 대한 정보 없이도((Model-Free) 충분히 Optimal Policy를 찾을 수 있다. 다양한 $$q(s,a)$$에 대해 충분히 정확하게 알기 위해 Exploring Start, Infinite Episode Sample 두 가지를 가정하며 이러한 점에서 **Monte Carlo ES(Exploring Start)**라고 한다. 알고리즘은 아래와 같다.

<img src="{{site.image_url}}/study/monte_carlo_es_algorithm.png" style="width:34em; display: block; margin: 0px auto;">

### 2.2. On-Policy & Off-Policy

Exploring Start를 사용하지 않으려면 Policy가 다양한 Action을 확률적으로 선택하도록 해야 한다. 이와 관련해 두 가지 방법, On-Policy와 Off-Policy가 있다. **On-Policy**는 Action의 선택에 사용되는 Policy(**Behavior Policy**)와 업데이트의 대상이 되는 Policy(**Target Policy**)가 같은 경우를 의미하고, **Off-Policy**는 다른 경우를 말한다. 

- On-Policy: Behevior Policy $$\mu(a \lvert s)$$ $$=$$ Target Policy $$\pi(a \lvert s)$$
- Off-Policy: Behevior Policy $$\mu(a \lvert s)$$ $$\neq$$ Target Policy $$\pi(a \lvert s)$$

이와 같은 구분이 중요한 이유는 Behavior Policy가 학습에 사용되는 Episode를 모으는 데에 사용되는 Action을 결정하는 Policy이므로 On-Policy는 Stochastic Policy여야 하지만 Off-Policy는 Target Policy가 Deterministic해도 되기 때문이다.

### 2.3. On-Policy: Policy Iteration with $$\epsilon$$-Greedy Policy

Stochastic Policy, 즉 어떤 State에서 Action이 확률적으로 결정되는 Policy의 대표적인 예로 $$\epsilon$$의 확률로 임의의 Action을 선택하는 $$\epsilon$$-Greedy Policy가 있다. 위의 Monte Carlo ES가 On-Policy이면서 Exploring Start를 적용하여 Policy Iteration을 수행하는 방법이었다면, $$\epsilon$$-Greedy Policy를 적용하여 Policy Iteration을 수행하는 것 또한 아래 알고리즘과 같이 가능하다고 할 수 있다. 이때 Policy는 확률적으로 어떠한 Action도 선택 가능한 **Soft Policy**여야 한다.

<img src="{{site.image_url}}/study/soft_policy_on_policy_algorithm.png" style="width:30em; display: block; margin: 0px auto;">

이와 관련하여 Greedy Policy가 아닌 $$\epsilon$$-Policy에 대해서도 Policy Improvement Theorem이 성립하는지는 다음과 같이 확인할 수 있다.

$$
\eqalign{
q_\pi(s, \pi'(s)) 
&= \Sigma_a \pi'(a \lvert s) q_\pi (s,a)\\
&= { \epsilon \over {\lvert A(s) \rvert}} \Sigma_a q_\pi (s, a) + (1 - \epsilon) \max_a q_\pi (s,a)\\
&\geqq { \epsilon \over {\lvert A(s) \rvert}} \Sigma_a q_\pi (s, a) + (1 - \epsilon) \Sigma_a { \pi (a \lvert s) - {\epsilon \over \lvert A(s) \rvert} \over 1 - \epsilon} q_\pi(s, a)\\
&= { \epsilon \over {\lvert A(s) \rvert}} \Sigma_a q_\pi (s, a) - { \epsilon \over {\lvert A(s) \rvert}} \Sigma_a q_\pi (s, a) + \Sigma_a \pi(a \lvert s) q_\pi (s, a) \\
&=\Sigma_a \pi(a \lvert s) q_\pi (s, a) \\
&= v_\pi (s)
}
$$

### 2.4. Off-Policy: Importance Sampling

Off Policy는 Behavior Policy와 Target Policy가 다르다고 했었다. 이는 어떤 Policy를 업데이트 하기 위해 다른 Policy가 모아준 경험을 사용하겠다는 것을 의미하고, 따라서 Behavior Policy가 경험할 확률이 높은 Trajectory를 Target Policy는 경험할 확률이 매우 낮을 수 있다. 다른 말로 하면 Trajectory에 대해 부여하는 중요도가 다를 가능성이 있다는 것이다. 따라서 Behavior Policy에 의해 모은 Trajectory를 그대로 사용하지 않고, 중요도를 반영하여 Target Policy를 업데이트 해야 한다.

이러한 문제를 수학적으로 표현해 보자면 어떤 Policy $$\pi$$가 어떤 Trajectory $$(A_t, S_{t+1}, A_{t+1}, ... S_T)$$를 경험할 확률은 다음과 같다.

$$
\Pi_{k=t}^{T-1} \pi(A_k \lvert S_k) p(S_{k+1} \lvert S_k, A_k)
$$

이를 활용하면 Trajectory $$(A_t, S_{t+1}, A_{t+1}, ... S_T)$$에 대해 $$\mu$$를 따를 때의 그럼직한 정도와 비교해 $$\pi$$를 따를 때의 그럼직한 정도가 얼마나 큰지를 다음과 같이 표현할 수 있다.

$$
\rho^T_t = {\Pi_{k=t}^{T-1} \pi(A_k \lvert S_k) p(S_{k+1} \lvert S_k, A_k) \over \Pi_{k=t}^{T-1} \mu(A_k \lvert S_k) p(S_{k+1} \lvert S_k, A_k)} = {\Pi_{k=t}^{T-1} \pi(A_k \lvert S_k) \over \Pi_{k=t}^{T-1} \mu(A_k \lvert S_k) } 
$$

이를 **Importance Sampling Ratio**라고 한다. **Importance Sampling**이란 알기 어려운 확률 분포를 알아내기 위해 상대적으로 알기 쉬운 확률 분포의 샘플과 두 확률 분포 간의 관계를 사용하는 방법이며, 여기서는 알기 쉬운 확률분포의 샘플이라고 할 수 있는 Behavior Policy $$\mu$$로 추출한 Episode들과 Behavior Policy $$\mu$$와 Target Policy $$\pi$$의 관계라고 할 수 있는 Importance Sampling Ratio를 사용하게 된다.

다음과 같이 몇몇 부가적인 수식을 사용하면

-  $$\tau(s)$$ : 전체 Time Step 동안 State $$s$$를 방문한 Time Step들의 집합
- $$T(t)$$: $$t$$시점 이후 첫 번째 Terminal State의 Time Step

다음과 같이 Value Function을 추정할 수 있다. 이러한 방법을 **Ordinary Importance Sampling**이라고 한다.

$$
V(s) = {\Sigma_{t \in \tau(s)} \rho_t^{T(t)} G_t \over \lvert \tau(s) \rvert}
$$

Importance Sampling Ratio에 가중치를 적용할 수도 있는데, 이를 **Weighted Importance Sampling**이라고 한다.

$$
V(s) = {\Sigma_{t \in \tau(s)} \rho_t^{T(t)} G_t \over \Sigma_{t \in \tau(s)} \rho_t^{T(t)} }
$$

당연하게도 Weighted Importance Sampling의 Variance가 더 작기 때문에 안정적으로 학습이 된다고 한다.

#### Apply Incremental Implementation

위의 Weighted Importance Sampling 식을 다음과 같이 Incremental 수식으로 바꾸어 표현할 수 있다. 여기서 $$W_i = \rho_t^{T(t)}$$라고 생각하면 된다.

$$
\eqalign{
V_{n+1} &= V_n + {W_n \over C_n} [G_n - V_n], \\
&\eqalign{\text{ where } &C_{n+1} = C_n + W_{n+1}\\ &C_0 = 0 \\ &n \geq 1
}}
$$

$$C_n$$이 누적 $$W_k$$라는 점을 생각하며 위 식을 다음과 같이 변경하면 보다 직관적으로 이해할 수 있다.

$$
V_{n+1} = V_n + {W_n \over C_n} [G_n - V_n] = V_n - {W_n \over C_n} V_n + {W_n \over C_n} G_n
$$

즉 현 시점의 $$W_n$$가 가지는 비율만큼 현재 예측하는 Expected Return 인$$V_n$$를 경험한 Return인 $$G_n$$으로 바꾸어 업데이트하겠다는 것이다.

<img src="{{site.image_url}}/study/incremental_monte_carlo_off_policy_algorithm.png" style="width:30em; display: block; margin: 0px auto;">

#### 2.5. Limitation of Off-Policy $$\epsilon$$-Greedy Algorithm

Off-Policy $$\epsilon$$-Greedy의 가장 큰 단점은 Non-Greedy Action의 비율이 높으면 높을 수록 학습에 오랜 시간이 소요된다는 것이다. 이는 업데이트를 하기 위해 전체 Episode가 필요하다는 Monte Carlo의 특징과 합쳐져 더욱 심화되며, Episode-by-Episode가 아닌 Step-by-Step으로 업데이트가 가능한 Temporal Difference 방법에서는 이러한 문제가 다소 완화된다.
