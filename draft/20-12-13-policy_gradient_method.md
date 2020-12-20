---
layout: post
title: Policy Gradient Method
category_num: 10
---

# Policy Gradient Method

- Sutton과 Barto의 Reinforcement Learning: An Introduction 2nd edition을 참고해 작성했습니다.  
- update at : 2020.12.13

## Policy Gradient

DQN을 비롯한 Q-learning 계열의 알고리즘들은 선택할 수 있는 Action들의 Value(q-value)를 추정하고, 그 값이 가장 큰 Action을 선택하는 **Action Value Method**이다. 이러한 Action Value Method 외에도 강화학습의 대표적인 방법론으로는 **Policy Gradient Method(PG)**가 있다. Policy Gradient라는 표현에서도 알 수 있듯이, PG에서는 State를 입력으로 받으면 Action을 출력하는 Policy를 직접 Parameterize 하고(**Parameterized policy**) 이것에 대한 Gradient를 구하여 업데이트한다.

$$
\pi(a \lvert s, \theta) = \text{Pr} \{ A_t = a \lvert S_t = s, \theta_t = \theta \}
$$

Actor-Critic Method와 같이 Parameterized Policy와 Value Function을 함께 업데이트하는 경우도 있는데, Policy를 일단 Parameterize하고 이것에 대한 Gradient를 구해 업데이트하면 PG로 분류한다(Value Function을 사용하는 이유 자체가 Policy를 더 잘 업데이트하기 위해서이다).

### Performance Measure

Policy를 직접 업데이트하기 때문에 정확한 Value를 얻을 수 있는 방향으로 업데이트하는 Q-learning과 달리 Policy 그 자체로 어떤 State가 들어왔을 때 얼마나 좋은 Action을 선택하는지를 기준으로 학습이 이뤄지게 된다. 이때 주어진 Policy Parameter가 얼마나 좋은지 측정하는 함수를 **Performance Measure**라고 하며, $$J(\theta)$$로 표기한다. 강화학습의 목표는 성능이 높은 Policy를 얻는 것이므로(Maximize) $$\theta$$를 업데이트 할 때에는 아래 식과 같이 **Gradient Ascent**가 적용된다.

$$
\theta_{t+1} = \theta_t + \alpha \widehat{\nabla J(\theta_t)}
$$

## Advantage of Policy Gradient

Policy Graident에서 Policy $$\pi$$는 다음과 같은 특성을 가진다.

1. $$\pi(a \lvert s, \theta)$$는 $$\theta$$에 대해 **미분 가능**하다.
2. $$\pi(a \lvert s, \theta)$$는 **Stochastic**하다. 즉 모든 State에 대해 어떤 Action도 선택할 확률이 0 또는 1이 될 수 없다.

Discrete Action Space에서 위의 두 가지 조건을 모두 만족하는 가장 기본적인 Policy는 다음과 같이 Softmax를 사용하는 것이다.

$$
\pi(a \lvert s, \theta) = {e^{h(s,a,\theta)} \over \Sigma_b e^{s, b, \theta}}
$$

여기서 $$h$$는 각각의 State-Action Pair $$(s,a)$$에 대한 선호의 크기를 의미하는데, 아래와 같이 딥러닝 모델로 표현할 수 있다. 참고로 $$x(s,a)$$는 State-Action Pair에 대한 Feature Vector이다.

$$
h(s,a,\theta) = \theta^\text{T} x(s,a)
$$

### Advantage in Exploration

위와 같은 Parameterized Policy를 사용하면 Q-learning과 비교해 볼 떄 우선 **탐색(Exploration)**의 관점에서 장점을 갖는다. 탐색을 위해 $$\epsilon$$-greedy를 사용하는 Q-learning에서는 적당한 $$\epsilon$$의 크기를 찾는 것이 어려울 뿐더러 학습이 진행되는 과정에서 $$\epsilon$$의 감소 폭 또한 적절하게 맞춰 주어야 한다. 반면 Policy Gradient에서는 모든 Action이 언제나 선택 가능하고, 그 확률 또한 최적이 되도록 학습하기 때문에 이러한 문제에서 자유롭다.

### Advantage as Stochastic Policy

두 번째 장점은 Parameterized Policy는 Optimal Policy가 Deterministic한 경우는 물론 Stochastic한 경우에도 잘 대처한다는 것이다. 어떤 State $$s$$에 대해 $$q(s,q)$$ value가 가장 큰 Action $$a$$를 선택하기 때문에 Q-learning에서의 Policy는 항상 Deterministic하다. 이는 몇몇 경우에서 성능을 제한하는 원인이 될 수 있는데, 특히 Poker처럼 다양한 경우의 수가 존재하는 경쟁적 게임에서 항상 동일한 전략을 구사하는 모델을 만들게 된다. 

반면 Policy Gradient에서는 항상 모든 Action을 선택할 가능성이 존재하기 때문에 기본적으로 Stochastic하나, Optical Policy가 Deterministic한 경우라면 학습 과정에서 Optimal Action의 선택 가능성이 sub-optimal과 비교해 무한히 높아지게 되고, 따라서 Deterministic Policy와 거의 유사해진다.

### Advantage in Convergence Guarantee

마지막으로 Parameterized Policy가 $$\epsilon$$-Greedy를 사용하는 Q-learning과 비교해 볼 때 이론적으로 보다 수렴성 보장이 확실하다는 장점이 있다. $$\epsilon$$-Greedy의 특성상 선택하는 Action이 급격하게(dramatically) 변화할 수 있고 이것이 안정적인 수렴에 방해 요인이 되지만, Paramterized Policy에서는 Action이 조금씩(Smooth)하게 변화하여 수렴성이 보다 강하게 보장된다는 것이다.

## Policy Gradient Theorem

이와 같은 여러 장점을 가지고 있지만 Paramterized Policy를 직접 업데이트하기 위해서는 Performance Measure $$J(\theta)$$에 대해 Gradient를 구해야 한다. **Policy Gradient Theorem**은 이것이 가능하다는 것을 보여주는 공식으로, 쉽게 말해 $$\nabla J(\theta)$$를 구하는 방법이라고 할 수 있다. 전개 과정에서는 Episodic Case, 즉 Terminal State가 존재하고, 매번 동일한 State $$s_0$$에서 시작한다는 것을 가정하는데, 구체적인 과정은 생략(Sutton, p325)하면 다음과 같다.

$$
\eqalign{
\nabla J(\theta) 
& = \nabla v_\pi (s_0) \\
&\varpropto \Sigma_s \mu(s) \Sigma_a q_\pi (s,a) \nabla \pi(a \lvert s, \theta)
}
$$

여기서 $$\mu (s)$$는 현재 Policy $$\pi$$의 **State Distribution**을 의미한다. Policy Gradient 알고리즘들은 모두 이 공식을 통해 Policy를 업데이트하게 되는데, 직접적으로 이를 사용하는 알고리즘이자 가장 전통적인 Policy Gradient 방법론으로 **REINFORCE**가 있다.

## REINFORCE

Policy Gradient Theorem에 따라 Gradient를 구한다고 할 때 가장 문제가 되는 것은 State Distribution $$\mu (s)$$이다. 즉, 현재의 Policy를 따를 때 어떤 State를 얼마나 자주 방문하는지 알아야 하는데, 정확한 값을 구하기 어려울 뿐더러 Policy가 업데이트되면 그 값이 바뀌게 되므로 매번 다시 구해야 한다.

$$
\nabla J(\theta) \varpropto \Sigma_s \mu(s) \Sigma_a q_\pi (s,a) \nabla \pi(a \lvert s, \theta)
$$

### $$s$$ to $$S_t$$

이러한 점 때문에 정확한 $$\mu$$를 구하는 것이 아니라 Episode를 진행하면서 방문하는 State의 분포로 대체하는 방법을 생각해 볼 수 있다. 쉽게 말해 Sampling으로 실제 Gradient를 추정하겠다는 것이다.

$$
\Sigma_s \mu(s) \Sigma_a q_\pi (s,a) \nabla \pi(a \lvert s, \theta) = E_\pi [\Sigma_a q_\pi (S_t, a) \nabla \pi (a \lvert S_t, \theta)]
$$

추정치를 사용하는 것이기 때문에 다소 부정확할 수 있지만 Step Size $$\alpha$$를 적게 주어 업데이트의 크기를 조절하는 식으로 부작용을 줄이는 것이 가능하다. Policy Gradient Themrem 식 부터 비례식이기 때문에 Step Size를 곱해주어도 크게 문제가 없다. 이러한 방법을 적용하면 업데이트 식을 다음과 같이 쓸 수 있다.

$$
\theta_{t+1} = \theta_t + \alpha \Sigma_a \hat q(S_t, a, w) \nabla \pi(a \lvert S_t, \theta)
$$

결과적으로 모든 State $$s$$가 아닌 특정 State $$S_t$$에 대한 식으로 대체되었다.

### $$a$$ to $$A_t$$

State 뿐만 아니라 Action에 대해서도 유사한 문제가 있는데, 위의 식을 계산하기 위해서는 어떤 state $$S_t$$에서 모든 Action $$a$$를 고려해야 한다는 점이다. 이 또한 State Distribution을 해결한 것처럼 Policy에서 Action을 Sampling하는 식으로 해결할 수 있다.

$$
\eqalign{
\nabla J(\theta) 
&\varpropto  E_\pi [\Sigma_a q_\pi (S_t, a) \nabla \pi (a \lvert S_t, \theta)]\\
& = E_\pi [\Sigma_a \pi (a \lvert S_t, \theta) q_\pi (S_t, a) { \nabla \pi (a \lvert S_t, \theta) \over \pi (a \lvert S_t, \theta)}]\\
& = E_\pi [q_\pi (S_t, A_t) { \nabla \pi (A_t \lvert S_t, \theta) \over \pi (A_t \lvert S_t, \theta)}]\\
& = E_\pi [G_t { \nabla \pi (A_t \lvert S_t, \theta) \over \pi (A_t \lvert S_t, \theta)}]\\
}
$$

두 번째 줄에서는 분자와 분모에 나란히 $$\pi (a \lvert S_t, \theta)$$를 곱해주고 있다. 그리고 세 번째 줄에서는 $$\Sigma_a \pi(a \lvert S_t, \theta)$$를 $$\pi$$에 대한 기대값으로 만들어주고, 이에 따라 $$A_t \backsim \pi$$로 $$a$$를 대체하고 있다. 마지막에는 $$q_pi (S_t, A_t) = E_\pi[G_t \lvert S_t, A_t]$$라는 점을 이용해 Return $$G_t$$로 식을 간단히 하고 있다. 최종적인 업데이트 식은 다음과 같이 구해지며, 이에 따라 업데이트하는 알고리즘을 **REINFORCE** 라고 한다.

$$
\theta_{t+1} = \theta_t + \alpha G_t { \nabla \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t)}
$$

### Update Size of REINFORCE

위의 REINFORCE 업데이트 식을 보면 Step Size $$\alpha$$외에 두 가지 스칼라 값, **Return Value $$G_t$$**와 어떤 State $$S_t$$에서 어떤 Action $$A_t$$를 선택할 확률 **$$\pi(A_t \lvert S_t, \theta)$$**에 의해 한 번에 업데이트되는 크기가 결정된다는 것을 알 수 있다. 업데이트의 방향은 $$\pi(A_t \lvert S_t, \theta)$$에 의해 결정되는데, 이는 쉽게 말하면 다음에 다시 $$S_t$$를 방문했을 때 다시금 $$A_t$$를 선택하도록 $$\theta$$를 업데이트하는 방향이라고 할 수 있다. 이러한 점에서 보면 Return이 높은 방향으로 강화하는 것은 직관적이다.

**$$\pi(A_t \lvert S_t, \theta)$$**에 반비례하여 업데이트 사이즈를 정하는 것은 자주 방문한다는 것 만으로 해당 방향으로 강화되는 것을 막기 위한 것으로 이해할 수 있다. 만약 이 분모 Term이 없다면 현재 Policy가 자주 선택한다는 것만으로도 해당 Action이 강화되어 sub-optimal에만 머무를 가능성이 높아진다.

### REINFORCE as Monte Carlo Algorithm

마지막으로 REINFORCE는 현재 Policy를 따랏을 때 얻어지는 Return $$G_t$$를 사용하여 업데이트한다. 따라서 한 번의 업데이트를 위해 전체 Episode를 진행해보고, 그 과정에서 얻은 Reward 값의 총합을 알아야 한다. 이러한 점에서 Monte Carlo Algorithm으로 분류된다. 알고리즘은 아래와 같다.

<img src="{{site.image_url}}/study/policy_gradient_method_reinforce_algorithm.png" style="width:36em; display: block; margin: 15px auto;">

**REINFORMCE**는 Step Size $$\alpha$$가 충분히 작고, 학습 과정에서 점차 줄어들면 이론적으로 수렴이 보장된다는 장점을 가지고 있다. 그러나 Monte Carlo를 사용한다는 점에서 Variance가 크고, 이로 인해 학습의 속도가 느리다는 문제를 가지고 있다. 하나의 Episode로 단 한 번만 업데이트가 가능하며, 업데이트가 되고 나면 기존의 Transaction은 모두 버려야 한다는 점에서 Sample Efficiency도 낮다.

## REINFORCE WITH BASELINE

REINFORCE가 가지는 High Variance 문제를 완화하는 가장 쉬운 방법 중 하나는 q-value에 **baseline $$b(s)$$** 개념을 도입하는 것이다.

$$
\nabla J(\theta) \varpropto \Sigma_s \mu(s) \Sigma_a (q_\pi (s,a) - b(s)) \nabla \pi(a \lvert s, \theta)
$$

REINFORCE Algorithm에 적용하면 다음과 같이 Return value에 baseline을 빼어주는 형태가 된다.

$$
\theta_{t+1} = \theta_t + \alpha (G_t - b(S_t)) { \nabla \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t)}
$$

참고로 Baseline $$b(s)$$는 Action $$a$$에 의해 결정되는 값만 아니면 된다. random variable이어도 된다. 가장 기본적인 방법은 Value Function $$\hat v(S_t, w)$$로 설정하는 것이다. 아래 알고리즘을 보면 $$w$$로 parameterized 되는 Value Function $$\hat v$$가 있고, 이를 Return Value $$G$$에 빼어준 값 $$\delta$$로 본래의 REINFORCE 업데이트 식에서의 $$G$$를 대체하고 있다.

<img src="{{site.image_url}}/study/policy_gradient_method_reinforce_with_baselint_algorithm.png" style="width:36em; display: block; margin: 15px auto;">

이와같이 Baseline을 적용하면 보다 빠르게 수렴한다고 한다.

## Actor-Critic

Actor-Critic은 Return $$G_t$$를 One-Step Return $$G_{t:t+1}$$로 대체하는 방법이라고 할 수 있다.

$$
\eqalign{
\theta_{t+1} 
&= \theta_t + \alpha (G_{t:t+1} - \hat v(S_t)) { \nabla \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t)}\\
&= \theta_t + \alpha (R_{t+1} + \hat v(S_{t+1}) - \hat v(S_t)) { \nabla \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t)}\\
}
$$

One-Step Return은 일반적인 Return과 비교하여 Variance는 줄어드는 대신 Bias가 발생하게 된다(MC vs TD).
