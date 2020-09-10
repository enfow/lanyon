---
layout: post
title: Temporal Difference
category_num: 6
---

# Temporal Difference

- Sutton의 2011년 책 Reinforcement Learning: An Introduction 2nd edition을 참고해 작성했습니다.  
- update at : 2020.09.10

## Introduction

Dynamic Programming(DP), Monte Carlo(MC) Method와 함께 **Temporal-Difference(TD) Leanring**는 강화학습의 대표적인 업데이트 방식이라고 할 수 있다. 그 중에서도 TD는 DP, MC와 비교해 많은 장점을 가지고 있으며 현재 유행하는 많은 강화학습 알고리즘에서 사용하는 방법론이다. 이와 관련하여 Sutton은 책에서 다음과 같이 표현하고 있다.

- "If one had to identify one idea as **central and novel to reinforcement learning**, it would undoubtedly be temporal-difference (TD) learning"

큰 틀에서 보면 세 가지 방법론은 모두 Policy Evaluation과 Policy Improvement를 반복하는 **Generalized Policy Iteration(GPI)**에 따라 동작하는데, 다만 그 과정에서 어떻게 Value Function을 추정할 것인가, Policy Evaluation을 수행하는 방식에서 차이가 있다고 할 수 있다.

## TD learning

TD는 DP, MC와 분명 다른 방법으로 업데이트하지만 책에서 언급하고 있듯이 두 방법 간의 Combination으로서 두 가지 방법의 장점을 취하고 있다. 결론부터 말하자면 TD는 DP와 같이 **BootStrapping**을 통해 업데이트 하기 때문에 Return을 알 필요가 없어 MC와 달리 에피소드가 끝나기를 기다리지 않아도 된다는 점에서 효율적이다. 그리고 MC와 같이 **Sampling**을 가정하기 때문에 DP와 달리 Model을 알지 못해도 된다는 점에서 자유롭다.

### BootStrapping

TD에서는 다음과 같이 Value Function을 업데이트 한다.

$$
V(s_t) \leftarrow V(s_t) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$

TD의 업데이트 식은 아래의 MC의 업데이트 식과 비교해서 보면 

$$
V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)]
$$

MC에서 전체 에피소드의 누적 Reward를 의미하는 $$G_t$$가 TD에서는 $$R_{t+1} + \gamma V(s_{t+1})$$로 대체되었다는 것을 알 수 있다. 한 마디로 $$s_t$$의 Value를 현재 받은 Reward와 다음 State $$s_{t+1}$$ Value 만을 사용하여 추정하겠다는 것이다. 이와 같이 True Value Function에 가까워지기 위해 Current Estimate Value Function으로 업데이트하는 것을 **BootStrapping**이라고 한다.

### Sampling

DP와 같이 BootStrapping을 사용하지만 본질적으로 TD의 업데이트 방식은 DP와 차이가 있다. DP의 업데이트 수식은 다음과 같이 얻어지는데

$$
\eqalign{
v_\pi(s) &= E_\pi[G_t \lvert S_t = s]\\
&=E_\pi [ \Sigma_{k=0}^\infty \gamma^k R_{t+k+1} \lvert S_t = s ]\\
&=E_\pi [ R_{t+1} + \gamma \Sigma_{k=0}^\infty \gamma^k R_{t+k+2} \lvert S_t = s ]\\
&=E_\pi [ R_{t+1} + \gamma v_\pi (S_{t+1})\lvert S_t = s ]\\
}
$$

Model에 대한 정보, 즉 State Transition과 Reward Function에 대해 알고 있다고 가정하는 DP에서는 기대값 $$E_\pi [ R_{t+1} + \gamma v_\pi (S_{t+1})\lvert S_t = s ]$$를 구할 수 있다. 하지만 TD의 경우 Model-Free를 가정하며, 다음 State에 대해 모르기 때문에 DP처럼 해결할 수 없다. 대신 MC처럼 기대값을 **Sampling**한 값으로 대체하게 된다.

참고로 MC, TD와 같이 State-Action Pair를 Sampling하여 Valu Function을 업데이트하는 것을 **Sample Backup**이라고 하고, DP와 같이 모든 가능성을 고려하여 업데이트 하는 것을 **Full Backup**이라고 한다.

### TD = Sampling of MC + Bootstrapping of DP

정리하자면 TD의 Target Value인 $$R_{t+1} + \gamma V(s_{t+1})$$는 다음 두 가지 측면에서 Estimation이라고 할 수 있다.

- Sampling
- Current Estimate Value Function

두 가지 모두에 대해 Estimation하기 때문에 Return을 계산하기 전에도, Model에 대해 알지 못해도 업데이트가 가능하다. TD의 업데이트 알고리즘은 아래와 같다.

<img src="{{site.image_url}}/study/td_algorithm.png" style="width:31em; display: block; margin: 0px auto;">

### Convergence of TD

두 가지에 대해 모두 추정한다는 점에서 TD Learning이 $$v_\pi$$로 수렴할 것인지 의문이 생길 수 있다. Step Size $$\alpha$$가 충분히 작거나 점진적으로 작아진다면 $$v_\pi$$로 수렴한다는 것이 증명되어 있다.

## TD Method for Control Problem

$$
V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$

TD Method를 Contorl 문제에 적용하는 기초적인 방법론으로 SARSA와 Q-learning이 있다. On-Policy인 SARSA에서는 Current Estimation $$V(s_t)$$과 Target Estimation $$V(s_{t+1})$$이 동일한 Policy에 따라 계산되지만, Off-Policy인 Q-learning에서는 서로 다른 Policy로 결정된다.

### SARSA : On-Policy TD

Model-Free의 문제를 풀기 위해서는 모든 State, Action Pair에 대한 Q Function 값을 추정해야 한다([Monte Carlo Method](<https://enfow.github.io/study/rl-study/2020/09/03/monte_carlo/>)). 따라서 위의 TD 업데이트 식을 Q function $$Q(s,a)$$에 맞춰 쓰면 다음과 같다.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
$$

위 식을 업데이트 하기 위해서는 $$(s_t, a_t, r_{t+1} +s_{t+1}, a_{t+1})$$이 필요하다. SARSA가 S,A,R,S,A인 이유가 여기에 있다. 이를 사용하는 SARSA 알고리즘은 다음과 같다.

<img src="{{site.image_url}}/study/td_sarsa_algorithm.png" style="width:34em; display: block; margin: 0px auto;">

위의 알고리즘에서 확인하고 넘어갈 만한 것으로는 (1) Q Function에 따라 $$s_t$$에서 $$a_t$$를 결정하고, $$s_{t+1}$$에서 $$a_{t+1}$$을 결정한다는 것과 (2) 매 Step에서 Policy가 업데이트 된다는 것이다. (3) 모든 State, Action Pair를 경험할 수 있도록 하기 위해 $$\epsilon$$-Greedy를 사용하고 있다는 점에서 On-Policy의 특성을 반영하고 있음을 알 수 있다. 참고로 첫째 줄에도 나와있지만 $$s_{t+1}$$이 Terminal State이면 $$Q(s_{t+1}, a_{t+1})$$은 0으로 계산된다.

### Q-learning : Off-Policy TD

TD Prediction 식을 거의 그대로 사용한 SARSA와 달리 Q-Laerning 식은 약간의 변화가 있다.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_a Q(s_{t+1},a) - Q(s_t, a_t)]
$$

Target을 계산할 때 사용되는 Term이 $$\gamma Q(s_{t+1}, a_{t+1})$$에서 $$\gamma \max_a Q(s_{t+1},a)$$로 바뀌었다. 이제는 $$s_t$$에서 Action $$a_t$$를 결정할 때와 $$s_{t+1}$$에서 $$a_{t+1}$$을 결정할 때 사용하는 Policy의 종류가 달라졌다. 보다 명확하게 $$s_t$$에서는 SARSA와 동일하게 $$\epsilon$$-Greedy Policy로 결정했지만 $$s_{t+1}$$에서는 항상 Q Value가 가장 클 때의 Action을 선택하고 있다. 이러한 점에서 Off-Policy라는 것이다.

<img src="{{site.image_url}}/study/td_q_learning_algorithm.png" style="width:34em; display: block; margin: 0px auto;">

SARSA와 비교해 볼 때 알고리즘이 더욱 단순해졌다. 이러한 점에서 Q-Learning을 강화학습에서 가장 중요한 발견 중 하나로 보기도 한다.