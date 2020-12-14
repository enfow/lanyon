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

Policy를 직접 학습하기 때문에 학습의 기준 또한 Value의 정확도가 아닌, Policy 자체의 성능이 되어야 한다. Policy Parameter에 따라 그것이 얼마나 좋은지 측정하는 함수를 **Performance Measure**라고 하며, $$J(\theta)$$로 표기한다. 강화학습의 목표는 성능이 높은 Policy를 얻는 것이므로(Maximize) $$\theta$$를 업데이트 할 때에는 아래 식과 같이 **Gradient Ascent**가 적용된다.

$$
\theta_{t+1} = \theta_t + \alpha \widehat{\nabla J(\theta_t)}
$$
