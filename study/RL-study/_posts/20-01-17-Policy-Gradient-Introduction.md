---
layout: post
title: Policy Gradient - REINFORCE & ACTOR-CRITIC
---

# Policy Gradient: REINFORCE & ACTOR-CRITIC

- Sutton의 2011년 책 Reinforcement Learning: An Introduction 2nd edition을 참고해 작성했습니다.  
- update at : 2020.01.17

## Policy Gradient

Policy Gradient(PG)란 value function을 이용해 action을 선택하는 것이 아니라 직접 action 값을 선택할 수 있도록 하는 policy parameter를 업데이트하는 방법을 말한다. 수학적으로는 다음과 같이 표현된다.

- $$\theta \in R^{d'}$$ : policy parameter

- $$\pi(a \lvert s, \theta) = Pr\{ A_t = a \lvert S_t = s, \theta_t = \theta \}$$ : policy

- $$J(\theta)$$ : objective function, policy parameter $$\theta$$를 사용할 때 성능 함수(performance measure)

위와 같은 표기법을 사용할 때 PG는 $$J(\theta)$$를 극대화하는 문제가 되며, 따라서 업데이트는 아래와 같이 $$J$$에 대해 gradient ascent를 진행하여 최적점을 찾는 방식으로 진행된다.

$$
\theta_{t+1} = \theta_t + \alpha \nabla \hat J(\theta_t)
$$

이때 $$\nabla \hat J(\theta_t)$$는 $$\theta_t$$에 있어 $$J$$의 기울기를 근사하는 기대값의 확률적인 추정치이다. 그리고 이러한 방법을 통해 최적화를 진행하면 업데이트 과정에서 value function을 사용하더라도 PG라고 부른다.

## Policy Gradient Theorem

PG 방법론의 핵심은 policy의 파라미터 $$\theta$$를 직접 학습하여 performance measure $$J$$가 극대화되도록 하는 것이다. 하지만 policy의 업데이트 방향을 찾는 것이 쉽지 않다. 왜냐하면 return을 최대로 하기 위해서는 현재 policy가 가지고 있는 state의 분포를 고려해야 하는데, agent로서는 이를 알기 어렵기 때문이다. 이러한 상황에서 항상 현재 state에서 최고의 reward를 갖는 action만 선택하도록 하면 장기적으로는 return을 증가시키지 못하고 빠르게 죽는 policy가 학습될 수 있다. 따라서 어떤 방향으로 policy를 업데이트한다고 할 때 그것이 실제로 성능을 높이는 방향인지 알기 어려운 것이 PG의 가장 큰 문제였다. Policy Gradient Theorem은 이러한 문제를 해결하기 위해 제시되었고, 이후 PG 방법론이 본격적으로 발전할 수 있었다.

Policy Gradient Theorem은 다음과 같다.

$$
\nabla J(\theta) \varpropto \Sigma_s \mu(s) \Sigma_a q_\pi (s,a) \nabla_\theta \pi(a \lvert s, \theta)
$$

여기서 $$\nabla J(\theta)$$는 $$\theta$$의 요소에 따른 $$J$$의 편미분 vector를 의미한다. 그리고 $$\mu$$는 on-policy distribution으로, $$\mu(s) = {n(s) \over {\Sigma_s} n(s) }, \ for \ all \ s \in S$$, 즉 전체 에피소드에서 어떤 state s를 방문할 확률의 기대값을 의미한다.

비례식으로도 충분한 이유는 어떤 수가 되더라도 step size $$\alpha$$에 의해 곱해진 값이 사용되기 때문이며, 그 비율은 step size를 조절하는 것과 동일한 효과이기 때문이다. 따라서 방향이 정확하고 크기는 비례하기만 하면 된다.

증명에서는 $$\nabla J(\theta) = \nabla V_\pi(s)$$에서 시작하며, 이를 위해 우선 $$\nabla V_\pi(s)$$를 product rule과 repeated rolling에 따라 전개하고 있다.

## REINFORCE

Policy Gradient Theorem의 우변은 target policy $$\pi$$를 따를 때 state들이 얼마나 자주 나타나는가에 따라 결정된다. 따라서 다음과 같이 표현할 수도 있다.

$$
\eqalign {
\nabla J(\theta) &\varpropto \Sigma_s \mu(s) \Sigma_a q_\pi (s,a) \nabla_\theta \pi(a \lvert s, \theta) \\
&= E_\pi [\Sigma_a q_\pi (S_t, a) \nabla_\theta \pi (a \lvert S_t, \theta)]
}
$$

즉 어느 state에 있을 것인가의 문제를 기대값으로 표현하는 것이다. 여기서 나아가 action $$a$$ 또한 $$A_t$$로 표현하기 위해 다음과 같이 식을 전개할 수 있다.

$$
\eqalign {
&= E_\pi [ \Sigma_a \pi(a \lvert S_t, \theta) q_\pi (S_t, a) { \nabla_\theta \pi (a \lvert S_t, \theta) \over \pi(a \lvert S_t, \theta) } ] \\
&= E_\pi [ q_\pi (S_t, A_t) { \nabla_\theta \pi (A_t \lvert S_t, \theta) \over \pi(A_t \lvert S_t, \theta) } ]
}
$$

식을 보면 $$\Sigma_a \pi(a \lvert S_t, \theta)$$를 $$A_t \backsim \pi$$로 대체하는 것을 알 수 있다. action 자체가 $$\pi$$에 의해 확률값으로 주어지기 때문에 성립한다. 끝으로 다음과 같이 return value G를 $$q_\pi(S_t \lvert A_t)$$에 대입해주게 되는데, 이를 REINFORCE 알고리즘이라고 한다.

$$
= E_\pi[ G_t { \nabla_\theta \pi (A_t \lvert S_t, \theta) \over \pi (A_t \lvert S_t, \theta_) } ]
$$

경우에 따라서는 $${d \over dx} \log f(x) = {f'(x) \over f(x)}$$를 적용하여 다음과 같이 간단하게 표현하기도 한다.

$$
\nabla_\theta J(\theta) = \Sigma_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \lvert s_t)G_t
$$

식의 업데이트 방식은 다음과 같다.

$$
\theta_{t+1} = \theta_t + \alpha G_t { \nabla_\theta \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t) }
$$

식을 풀이하자면, 한 번의 업데이트로 return $$G_t$$와 어떤 state에서 어떤 action을 선택할 확률의 gradient를 그 확률값으로 나눈 값인 $$ { \nabla_\theta \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t) }$$ 간의 곱이라고 할 수 있다. $$ { \nabla_\theta \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t) }$$는 업데이트의 방향을 결정하는 것으로, 다음에 어떤 state $$S_t$$를 다시 만나게 되면 action  $$A_t$$를 선택할 확률을 높여주는 방향을 의미한다. 만약 전체 episode의 return이 크다면 확률 또한 크게 높여줄 것이라고 기대할 수 있다.

REINFORCE 알고리즘은 t 시점에서의 전체 episode의 return을 사용한다. 이러한 점 때문에 Monte Carlo 알고리즘의 특성을 가진다고 할 수 있다. 그리고 episode가 한 번 끝나기 전에는 업데이트를 할 수 없다. 끝으로 현재의 policy $$\pi$$에 의해 얻어진 episode로만 업데이트를 한다는 점에서 on-policy적 특성을 갖는다.

이러한 REINFORCE의 특성은 곧바로 알고리즘의 단점으로 이어진다. 우선 전체 episode가 끝난 뒤 그 return을 이용한다는 점에서 episode의 길이 등에 따른 return의 variance가 매우 높다는 문제가 있다. 또한 return 값을 추정하는데 알고리즘적인 bias가 거의 없다. 이 점은 장점 같지만 별도의 exploration을 하지 않기 때문에 local minimum에 빠질 가능성이 매우 높다는 문제로 이어진다. 마지막으로 다른 Monte Carlo 방법들 처럼 학습의 속도가 매우 느리다.

이러한 문제들 때문에 실제로 REINFORCE 알고리즘을 실험해보면 수렴이 잘 되지 않는다. 이를 극복하기 위해 나온 방법론 중 하나로 Actor-Critic이 있다.

## Actor-Critic

Actor-Critic 방법은 말 그대로 actor와 critic 두 부분으로 이뤄진 알고리즘 구조를 말한다. 여기서 actor는 action을 선택하는 역할을 하며, critic은 actor에 의해 선택된 action을 평가한다. crtic을 구현하는 방법은 다양한데, 무엇이 되더라도 critic은 선택된 action의 결과로 얻은 next state를 평가하고, 이것이 기대보다 좋은지 나쁜지를 구하여 actor에게 해당 action을 선택하는 것이 좋은지 나쁜지 알려주는 역할을 한다.

이러한 Actor-Critic은 critic 구조를 통해 return $$G$$를 대체하여 REINFORCE 등과 같은 전통적인 PG 알고리즘이 가지고 있는 high variance 문제와 업데이트 속도의 문제 등을 해결하려고 하는 접근 방법이라고 할 수 있다.

가장 기본적인 Actor-Critic 구조로 critic에 state value function을 사용하여 TD error를 이용하는 One-Step Actor-Critic이 있다. One-Step Actor-Critic은 이름과 같이 전체 episode가 아닌 한 step 이후의 결과만 사용한다.

One-Step Actor Critic의 업데이트 식은 다음과 같이 구해진다.

$$
\eqalign {
\theta_{t+1} &= \theta_t + \alpha (G_{t:t+1} - \hat v(S_t, w)) { \nabla_\theta \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t) } \\
&= \theta_t + \alpha(R_{t+1} + \gamma \hat v (S_{t+1}, w) - \hat v (S_t, w)) { \nabla_\theta \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t) } \\
&= \theta_t + \alpha \delta_t { \nabla_\theta \pi (A_t \lvert S_t, \theta_t) \over \pi (A_t \lvert S_t, \theta_t) }
}
$$

여기서 첫 번째 식의 $$\hat v(S_t, w)$$는 state value function을 baseline으로 사용하는 것을 의미한다. 이를 적용한 알고리즘을 REINFORCE with baseline이라고 하며, 일반적인 REINFORCE보다 빠르게 수렴한다고 한다. 그리고 $$G_{t:t+1}$$란 1 step 뒤의 return만 보겠다는 것을 의미한다. 이는 두 번째 식에서 $$R_{t+1} + \gamma \hat v (S_{t+1}, w)$$로 표현되며 baseline과 합쳐져 TD-error $$\delta_t$$가 된다.

이렇게 전체 return을 보지 않고 한 step 뒤만 보게 되면 전체 episode를 구하지 않고 단일 transaction $$(s, a, r, s')$$만으로도 업데이트가 가능해진다. 이러한 점에서 매 step 업데이트가 가능해지고, 한 스텝 뒤의 reward만 사용하므로 variance가 줄어든다.
