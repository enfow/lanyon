---
layout: post
title: TRPO) Trust Region Policy Optimization
---

# 논문 제목 : Trust Region Policy Optimization

- John Schulman, Sergey Levine 등
- 2015
- [논문 링크](<https://arxiv.org/abs/1502.05477>)
- 2020.03.01 정리

## Summary

- PG의 가장 큰 문제 중 하나는 정확한 업데이트 방향을 알기 위해서는 매우 많은 sample이 필요하다는 것이다.
- **conservative policy iteration**(Kakade&Langford)처럼 제한된 범위 내에서 업데이트를 하면 적어도 성능이 유지되는 업데이트를 반복적으로 할 수 있다.
- **TRPO**는 mixture policy가 아닌 **stochastic policy**에 conservative policy iteration 방법론을 적용하고 있으며, 그 과정에서 **trust region update**와 **monte carlo estimation**을 사용한다.

## Policy Update

강화학습 알고리즘의 목표는 **expected return**을 극대화하는 policy를 찾는 것이다. 즉 아래 $$\eta$$ 식을 극대화하는 것이다.

$$
\eqalign{
&\text{maximize} \ \eta \\
&\eqalign{
\text{where} \ &\eta(\pi) = E_{s_0, a_0 ...} [\Sigma_{t+0}^\infty \gamma^t r(s_t)] \\
&\ s_0 \backsim \rho_0 (s_0), \ a_t \backsim \pi(a_t \lvert s_t), \ s_{t+1} \backsim P(s_{t+1} \lvert s_t, a_t)
}
}
$$

이와 관련하여 Kakade&Langford는 2002년 논문 Approximately Optimal Approximate Refinforcement Learning에서 **Conservative Policy Iteration**을 제시했었다. TRPO의 알고리즘은 이 논문에서 출발하고 있다.

## Before TRPO - Conservative Policy Iteration

Kakade&Langford는 자신의 논문에서 policy gradient 방법은 정확한 업데이트 방향(gradient)를 구하는 데에 너무 많은 sample을 확보해야 하며, 심지어 한 번 policy를 업데이트한 후에는 더 이상 과거에 사용한 sample을 사용할 수 없기 때문에 알고리즘적 비용이 너무 많이 든다고 한다. 이러한 문제를 해결하기 위해 업데이트의 크기를 제한하면서 policy의 expected return을 지속적으로 향상시키는 방법으로 Conservative Policy Iteration을 제시하고 있다.

### Expected return and Advantage

$$
\eta(\tilde \pi) = \eta(\pi) + E_{s_0, a_0, ... \backsim \tilde \pi} [\Sigma_{t=0}^\infty \gamma^t A_\pi (s_t, a_t)]
$$

기존의 policy $$\pi$$에서 새로운 policy $$\tilde \pi$$로 업데이트되었다고 할 때 이 때 expected return의 변화는 모든 시점에서 $$\gamma$$-discounted advantage $$A_\pi$$ 총합의 기대값이라고 할 수 있다. 즉 이 기대값이 음수이면 return이 줄어들고, 양수이념 return이 늘어나는 것으로 예상하는 것이다.

이때 state visitation frequency $$\rho_\pi (s)$$를 아래와 같이 정의하면

$$
\rho_\pi (s) = P(s_0 = s) + \gamma P(s_1 = s) + \gamma^2 P(s_2 = s) + ...
$$

기대값을 풀어 다음과 같이 전개할 수 있다.

$$
\eqalign{
    \eta(\tilde \pi) &= \eta(\pi) + \Sigma_{t=0}^\infty \Sigma_s P(s_t = s \lvert \tilde \pi) \Sigma_a \tilde \pi (a \lvert s) \gamma^t A_\pi (s,a) \\
    &= \eta(\pi) + \Sigma_s \Sigma_{t=0}^\infty \gamma^t P(s_t = s \lvert \tilde \pi) \Sigma_a \tilde \pi (a \lvert s) A_\pi (s,a) \\
    &= \eta(\pi) + \Sigma_s \rho_{\tilde \pi}(s) \Sigma_a \tilde \pi (a \lvert s) A_\pi (s,a) \\
}
$$

위의 식에서 $$\Sigma_a \tilde \pi (a \lvert s) A_\pi (s,a) \geqq 0$$ 이 성립하면 $$\eta$$가 증가하는 것을 보장할 수 있다. 이렇게 policy를 업데이트하는 대표적인 방법이 deterministic policy $$\tilde \pi(s) = \arg \max_a A_\pi (s,a)$$를 사용하는 policy iteration 이며, 이렇게 하면 optimal policy에 수렴할 수 있다.

하지만 이를 곧바로 적용하는 것에는 문제가 있다. estimation과 approximation에 따른 오차의 가능성 때문에 negative value가 포함될 수 있고, $$\rho_{\tilde \pi}(s)$$를 사용한다는 점에서 새로운 policy가 어떤 state에 자주 가는지 곧바로 알기 어렵기 때문에 즉각적인 업데이트도 불가능하다.

### Local approximation

이러한 문제를 해결하기 위해 아래와 같은 식을 도입하고 있으며, 이를 **local approximation** 이라고 한다.

$$
L(\tilde \pi)= \eta(\pi) + \Sigma_s \rho_{\pi}(s) \Sigma_a \tilde \pi (a \lvert s) A_\pi (s,a)
$$

$$\rho_{\tilde \pi}(s)$$가 $$\rho_{\pi}(s)$$로 바뀌었다. 즉 policy가 업데이트되어 $$\rho$$가 바뀌었음에도 이를 사용하지 않고 이전 policy의 $$\rho$$를 그대로 사용하겠다는 것이다.

하지만 이와 같은 근사를 사용하기 위해서는 다음과 같은 조건을 만족해야 한다.

$$
L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})\\
\nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta) \lvert_{\theta = \theta_0} = \nabla_\theta \eta(\pi_\theta) \lvert_{\theta = \theta_0}
$$

즉 충분히 작은 업데이트 크기를 정하게 되면 $$L_{\pi_\theta}$$를 기준으로 업데이트하는 것만으로도 $$\eta$$의 향상을 보장할 수 있다.

### Conservative Policy Iteration

위의 이론적 논의를 바탕으로 Kakade&Langford는 다음과 같은 **mixture policy update** 방법을 제시한다.

$$
\pi_{\text{new}}(a \lvert s) = (1 - \alpha) \pi_{\text{old}}(a \lvert s) + \alpha \pi ' (a \lvert s)
$$

즉 기존 policy $$\pi_{\text{old}}(a \lvert s)$$와 업데이트 방향이 되는 policy $$\pi ' (a \lvert s)$$의 가중평균으로 새로운 policy $$\pi_{\text{new}}(a \lvert s)$$를 정하는 것이다.

이 경우 다음과 같은 업데이트의 lower bound를 갖는다고 한다.

$$
\eta(\pi_{\text{new}}) \geqq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - {2 \epsilon \gamma \over (1-\gamma)^2} \alpha^2 \\
\text{where} \ \epsilon = \max_s \lvert E_{a \backsim \pi'(a \rvert s)} [A_\pi (s,a)] \lvert
$$

TRPO는 이와같은 Kakade&Langford의 Conservative Policy Iteration를 개선한 것이라고 할 수 있다.

## Trust Region Policy Optimization

### Monotonic Improvement Guarantee for General Stochastic Policy

Conservative Policy Iteration에서는 새롭게 구한 policy와 기존 policy를 적당한 크기로 가중평균하고 있다. 하지만 이와 같이 mixture를 통해 새로운 policy를 구하는 것은 stochastic policy에는 적용하기 어렵다. TRPO는 stochastic policy에서도 Conservative Policy Iteration와 유사한 방법론을 적용하여 policy를 업데이트하는 방법이라고 할 수 있다.

이를 위해 TRPO 논문에서는 $$\alpha$$와 $$\epsilon$$을 다음과 같이 재정의한다.

$$
\eqalign{
&\alpha = D_{\text{TV}}^{\max}(\pi_{\text{old}}, \pi_\text{new})\\
&\epsilon = \max_{s, a} \lvert A_\pi (s,a) \rvert
}
$$

$$\alpha$$는 두 policy 간의 차이가 얼마나 큰지를 의미하는데, 이때 둘 간의 차이를 TVD(total variation divergence)로 구하고 있음을 알 수 있다.

이러한 가정에 따르면 다음과 같이 새로운 lower bound를 설정할 수 있다.

$$
\eqalign{
    &\eta(\pi_{\text{new}}) \geqq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - {4 \epsilon \gamma \over (1 - \gamma)^2} \alpha^2 \\
    & \text{where} \ \epsilon = \max_{s,a} \lvert A_\pi (s,a) \rvert
}
$$

그런데 TVD는 두 분포 간의 차이를 계산할 때 자주 사용하는 KLD와 다음과 같은 관계를 가지고 있다.

$$
D_{TV}(p \| q)^2 \leqq D_{KL}(p \| q)
$$

이러한 특성을 $$\alpha^2$$에 적용하고, 상수들을 $$C$$로 치환하면 다음과 같이 식을 정리할 수 있다.

$$
\eqalign{
    &\eta(\pi_{\text{new}}) \geqq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - C D_{KL}^{\max}(\pi_{\text{old}, \pi_{\text{new}}})\\
    & \text{where} \ C = {4 \epsilon \gamma \over (1 - \gamma)^2}
}
$$

여기서 $$L_{\pi_{\text{old}}}(\pi_{\text{new}}) - C D_{KL}^{\max}(\pi_{\text{old}, \pi_{\text{new}}})$$를 극대화하는 $$\pi_{\text{new}}$$를 다음 policy로 업데이트하면 지속적인 성능 개선이 보장되는 policy update가 가능하다.

이를 적용하여 다음과 같은 algorithm을 도출할 수 있다.

<img src="{{site.image_url}}/paper-review/trpo_algorithm.png" style="width: 28em">

여기까지가 TRPO의 이론적인 업데이트 방식이라고 할 수 있다. 하지만 이를 곧바로 적용하기에는 연산량 등을 고려하여 보다 구체화해야 할 부분들이 남아있다.

### Trust region contraint Not Penalty

위에서는 $$C D_{KL}^{\max}(\pi_{\text{old}}, \pi_{\text{new}})$$의 크기만큼 penalty를 부여하는 방식을 사용하고 있다. 하지만 $$C$$에 포함되어 있는 $$\epsilon$$의 정의 $$\max_{s, a} \lvert A_\pi (s,a) \rvert$$를 고려하면 결국 매 state에서 새롭게 구해야 한다. 이를 대신하여 충분히 작은 $$\delta$$ 값 내에서만 변화할 수 있도록 constraint를 주는 방법을 생각해 볼 수 있다.

$$
\eqalign{
    &\text{maximize}_{\theta} \ L_{\theta_{\text{old}}}(\theta)\\
    &\text{subject to} \ D_{\text{KL}}^{\text{max}} (\theta_\text{old}, \theta) \leqq \delta
}
$$

하지만 이 경우에도 state에 따라 constraint가 늘어나는 문제를 가지고 있다. 이러한 문제를 해결하기 위해 $$D_{KL}^{\max}$$를 다음과 같은 state visitation frequency에 대한 기대값으로 대체한다.

$$
D_{KL}^{\rho}(\theta_1, \theta_2) := E_{s ~ \rho} [D_{KL} (\pi_{\theta_1}(\cdot \lvert s) \| \pi_{\theta_2}(\cdot \lvert s))]
$$

최적화 식에 적용하면 다음과 같다.

$$
\eqalign{
    &\text{maximize}_{\theta} \ L_{\theta_{\text{old}}}(\theta)\\
    &\text{subject to} \ D_{\text{KL}}^{\rho_{\theta_{old}}} (\theta_\text{old}, \theta) \leqq \delta
}
$$

heuristic approximation이기 때문에 항상 최적은 아닐 수 있지만 효율적으로 적당한 업데이트 크기를 정하는 데에 있어 도움이 된다.

### Monte Carlo approximation

위의 최적화 식에서 $$L$$을 전개하면 다음과 같다.

$$
\eqalign{
    &\text{maximize}_{\theta} \ \Sigma_s \rho_{\theta_{old}}(s) \Sigma_a \pi_{\theta}(a \lvert s) A_{\theta_{old}}(s,a)\\
    &\text{subject to} \ D_{\text{KL}}^{\rho_{\theta_{old}}} (\theta_\text{old}, \theta) \leqq \delta
}
$$

아직 많이 남은 것을 알 수 있다. 위의 식을 그대로 해결하려면 모든 state, action에 대한 값을 구해야 하므로 연산량이 매우 커진다. 따라서 Monte Carlo를 적용해 근사하는 방법을 생각할 수 있다.

보다 구체적으로 논문에서는 다음 세 가지를 도입하였다고 한다.

- Summation $$\Sigma_s \rho_{\theta_{old}}(s)[...]$$ $$\Rightarrow$$ Expecation $${1 \over 1 - \gamma} E_{s \backsim \rho_{\theta_{old}}}[...]$$
- Advantage $$A_{\theta_{old}}$$ $$\Rightarrow$$ Q-value $$Q_{\theta_{old}}$$
- importance sampling estimator $$q$$

sampling distrubution $$q$$를 통해 loss function은 다음과 같이 정의되고,

$$
\Sigma_a \theta_{\pi}(a \lvert s_n) A_{\theta_{old}}(s_n, a) = E_{a \backsim q}[{\pi_{\theta}[a \lvert s_n] \over q(a \lvert s_n)} A_{\theta_{old}}(s_n, a)]
$$

다른 두 변경사항을 적용한 최적화 식은 다음과 같다.

$$
\eqalign{
    &\text{maximize}_{\theta} \ E_{s \backsim \rho_{\theta_{old}}, a \backsim q} [{\pi_{\theta}[a \lvert s_n] \over q(a \lvert s_n)} Q_{\theta_{old}}(s_n, a)]\\
    &\text{subject to} \ E_{s \backsim \rho_{\theta_{old}}} [D_{\text{KL}} (\pi_{\theta_{old}}(\cdot | s) \| \pi_\theta (\cdot \lvert s))] \leqq \delta
}
$$

### Two scheme for sampling

마지막으로 한 가지 남은 것이 있다면 Monte Carlo를 통해 위의 최적화 식에서 expectation을 sampling으로 구하는 것이다. 이때 sampling 방법과 관련해 **single path**와 **vine** 두 가지가 있다.

#### 1. single path

- 개개의 episode trajectory를 그대로 사용하는 방법
- 전형적인 policy gradient estimation 방법

#### 2. vine

- rollout을 통해 하나의 state에서 여러 action을 취해보는 방법
- single path와 비교해 variance가 낮다는 장점
- real world에 적용하기 어렵다는 단점

### TRPO PROCESS

TRPO는 다음과 같은 순서로 policy를 업데이트한다.

1. single path 또는 vine 방식을 통해 state-action pair를 수집한다. 이렇게 수집된 state-action pair로 Monte Carlo estimation을 실시한다.
2. 최적화 식의 objective function과 constraint를 추정한다.

    $$
    \eqalign{
        &\text{maximize}_{\theta} \ E_{s \backsim \rho_{\theta_{old}}, a \backsim q} [{\pi_{\theta}[a \lvert s_n] \over q(a \lvert s_n)} Q_{\theta_{old}}(s_n, a)]\\
        &\text{subject to} \ E_{s \backsim \rho_{\theta_{old}}} [D_{\text{KL}} (\pi_{\theta_{old}}(\cdot | s) \| \pi_\theta (\cdot \lvert s))] \leqq \delta
    }
    $$

3. 2에서 계산된 값에 따라 policy parameter $$\theta$$를 업데이트 한다. 이때 line search를 따르는 conjugate gradient algorithm을 사용한다. 이는 gradient 자체를 구하는 방법보다는 약간 더 비싼 방법이라고 한다.
