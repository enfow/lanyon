---
layout: post
title: Bayesian Rule
category_num: 3
---

# Bayesian Rule

- update date : 2020.02.20, 2020.09.14

## Bayesian Rule

베이즈 정리는 Prior, Likelihood, Posterior 간의 관계를 표현하는 식으로, 아래와 같다.

$$
P(A \lvert B) = {P(B \lvert A) P(A) \over P(B)}
$$

여기서 $$P(A)$$를 **Prior**, $$P(B \lvert A)$$를 **Likelihood** 그리고 $$P(A \lvert B)$$를 **Posterior**라고 한다.

## Prior, Likelihood, Posterior

Prior, Likelihood, Posterior는 베이지언 확률에서 가장 기초가 되는 개념으로, 우리말로 사전 확률, 가능도(우도), 사후 확률로 불린다.

- **Prior**: 원인 사건이 발생할 확률
- **Likelihood**: 원인 사건이 발생했을 때 결과 사건이 발생할 확률
- **Posterior**: 결과 사건이 발생했을 때 원인 사건이 발생했을 확률

위와 같은 정의만 보아서는 무엇을 의미하는지 추상적으로만 들린다. 다음과 같은 구체적인 예시를 생각하면 이해하는 데에 도움이 된다.

### Show Me the Money

<img src="{{site.image_url}}/study/bayesian_money_in_the_box.jpg" style="width:40em; display: block; margin: 2px auto;">

여러 개의 케리어 중에 중에 돈 뭉치가 들어있는 것이 숨겨져 있다고 하자. 공항 보안검색대 직원은 수많은 케리어 중에서 돈 뭉치가 들어 있는 것을 찾아내야 한다. 이때 아래 각각의 확률 함수의 의미는 다음과 같다고 하자.

- $$P(\text{money} = \text{True})$$: 케리어에 돈이 들어 있을 확률
- $$P(\text{alarm} = \text{True})$$: 보안 검색대에 알람이 울릴 확률
- $$P(\text{alarm} = \text{True} \lvert \text{money} = \text{True})$$: 돈이 들어 있는 캐리어에 대해 보안 검색대 알람이 울릴 확률

이때 조건부 확률 $$P(X \lvert Y)$$는 그 정의에 따라 다음과 같이 쓸 수 있다.

$$
P(X|Y) = {P(X, Y) \over P(Y) }
$$

이를 돈다발 문제에 적용하면 $$P(\text{alarm} = \text{True} \lvert \text{money} = \text{True})$$는 다음과 같이 풀어 쓸 수 있다.

$$
\eqalign{
P(\text{alarm} = \text{True}\lvert \text{money} = \text{True})
&= {P(\text{money} = \text{True}, \text{alarm} = \text{True}) \over P(\text{money} = \text{True})} \\
&= {P(\text{money} = \text{True} \lvert \text{alarm} = \text{True}) P(\text{alarm} = \text{True}) \over P(\text{money} = \text{True})}
}
$$

이를 Prior, likelohood, Posterior로 분리해 그 의미를 생각해보면 다음과 같다.

- **Prior**: 어떤 케리어에 대해 알람이 울릴 확률
- **Likelihood**: 알람이 울린 케리어에 돈다발이 들어있을 확률
- **Posterior**: 돈다발이 들어 있는 케리어에서 알람이 울릴 확률

## Machine Learning and Bayesian Rule

여기서 문제가 있다면 보안 검색대가 완벽하지 못해 돈다발이라고 판단한 것 중에 돈이 없는 경우가 많아 항의가 많이 들어온다는 것이다. 이를 해결하기 위해 보안 검색대의 성능을 개선해야 해야하는데, 가장 쉽게 생각할 수 있는 것이 경험을 토대로 학습하는 것이다.

### Learning with Experience

만약 10개의 케리어 $$X = {x_1, x_2, ... x_{10}}$$가 있고 그 중 실제로 돈다발이 들어 있는 케리어는 $$x_1, x_2, x_9$$라고 하자. 이때 보안 검색대에서 전체 케리어 중 $$x_2, x_5$$에 돈다발이 들어 있다고 판단했다면, **Prior**는 $$P(alarm) = 0.2$$가 된다. 그런데 돈다발이 들어 있다고 판단된 케리어 두 개를 열어보니 $$x_2$$에는 돈다발이 있었지만, $$x_5$$에는 사과가 있었다. 따라서 **Likelihood** $$P(money \lvert alarm)$$는 0.5이다.

Prior와 Likelihood를 모두 구했지만 Posterior를 구하기 위해서는 분모 $$p(money)$$를 구해야 한다.

$$
P(money) = P(money \lvert alarm) P(alarm) + P(money \lvert \backsim alarm) P(\backsim alarm)
$$

위 식을 계산하면 $$P(money) = 0.3$$이 되는 것을 알 수 있다.

$$
{P(money \lvert alarm) P(alarm) \over P(money)} = { 0.5 * 0.2 \over 0.3} = {1 \over 3}
$$

결과적으로 **Posterior** $$P(alarm \lvert money)$$는 $$1 \over 3$$를 Prior로 업데이트하여 성능을 개선할 수 있는데 이를 학습이라고 할 수 있다. 전체 10개의 케리어 중 3개에 돈이 있었으므로, 실제 True Prior가 0.3임을 감안하면 학습의 결과로 Optimal 값에 보다 가까워진 것이라고 할 수 있다. 조금 어렵게 말한 감이 있지만 다음 예측을 할 때에는 이전에 배운 Posterior $$P(alarm \lvert money)$$를 고려하여 Prior $$P(alarm)$$를 결정하면 보다 정확도를 높일 수 있다는 것이다.
