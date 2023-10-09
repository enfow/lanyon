---
layout: post
title: Multi-armed Bandit
category_num: 2
---

# Multi-armed Bandit

- Sutton의 2011년 책 Reinforcement Learning: An Introduction 2nd edition을 참고해 작성했습니다.

## Keywords

본문에서 다루고 있는 강화학습의 주요 키워드들은 다음과 같다.

- **Evaluation & Instruction**: 강화학습은 어떤 Action이 좋다는 지시(Instruction)를 학습하는 것이 아니라 어떤 Action이 얼마나 좋은지 정확하게 평가(Evaluation)하는 것을 학습한다.
- **Action Value**: 어떤 Action을 수행했을 때 얻을 수 있는 Reward를 의미한다.
- **Greedy Action**: 선택 가능한 Action 중 Action Value가 가장 큰 것을 말한다.
- **Exploitation & Exploration**: Exploitation은 현재 추정 Action Value가 가장 큰 Action을 선택하는 것을, Exploration은 그 이외의 Action을 선택하고 학습하는 것을 말한다.
- **$$\epsilon$$-Greedy Method**: $$\epsilon$$만큼 Exploration을, $$1 - \epsilon$$만큼 Exploitation을 수행하는 방법을 말한다.

## Evaluation Aspact of Reinforcement Learning

Sutton은 책에서 강화학습이 머신러닝의 다른 방법들과 가지는 가장 큰 차이점을 다음과 같이 표현하고 있다.

- It uses training information that **evaluates** the actions taken rather than **instructs** by giving correct actions

여기서 두 가지 키워드 Evaluation과 Instruction이 나오는데 **Instruction**은 무엇이 가장 좋은 것인지, 무엇을 해야하는지 미리 정해져있고 그것에 따라 학습을 진행하는 것을 의미한다. 반면 **Evaluation**은 자신이 선택한 것이 좋다면 얼마나 좋고, 나쁘다면 얼마나 나쁜지에 관한 것에 대해 알아가도록 학습을 진행하는 것을 말한다.

예를 들어 대표적인 지도학습 문제인 Image Classification은 강아지 사진은 0번, 고양이 사진은 1번 등과 같이 label을 미리 정해두고 그것에 따라 학습하게 된다. 이러한 점에서 Instruction이 존재한다고 할 수 있다. 반면 강화학습에서는 강아지 사진을 보았을 때 0을 선택한 경우와 1을 선택한 경우를 모두 경험해보고 그것에 따른 보상을 평가하는 방향으로 학습을 진행한다. 이러한 점에서 Evaluation의 특성을 가진다고 하는 것이다.

## Multi-armed Bandit Problem

**Multi-armed Bandit Problem**은 강화학습의 Evaluation 특성을 잘 보여주는 문제다. 책에서는 $$n$$-Armed Bandit이라고 하여 $$n$$개의 선택 가능한 Action이 주어져 있을 때 정해진 시간 내에 Total Reward를 극대화할 수 있는 방법을 찾는 문제로 설명한다. 일반적인 강화학습 문제에서는 현재 State에 따라 그에 맞춰 Action을 결정하면 Reward를 받는 것과 동시에 새로운 State가 주어지게 되는데 Multi-armed Bandit에서는 State 개념이 존재하지 않아 항상 동일한 State를 가정하고 Action을 반복적으로 수행하는 상황을 가정한다.

$$n$$-Armed Bandit 문제의 대표적인 예시가 슬롯 머신이다. 카지노에 $$n$$개의 슬롯 머신이 있고 각각의 슬롯 머신에서 잭팟이 터질 확률이 모두 다르다고 하자. 그렇다면 어떤 슬롯 머신을 동작시키느냐에 따라 받을 수 있는 당첨금의 기대값이 달라지게 될 것이다. Multi-armed Bandit 문제는 어떤 슬롯 머신의 기대값이 가장 큰 지 모르는 상황에서 주어진 시간동안 한 번에 하나의 슬롯 머신만 동작시킬 때 당첨금을 극대화할 수 있는 방법을 찾는 것과 같다.

### Action Value Estimation

위와 같은 슬롯 머신 예시는 다음과 같이 강화학습의 개념들과 매칭할 수 있다.

- 당첨금 : Reward
- 슬롯 머신 : Action

여기서 문제는 각 슬롯 머신이 가지는 당첨금의 기대값을 알지 못한다는 점이다. 왜냐하면 당첨금의 기대값을 알고 있다면 그것이 가장 큰 슬롯 머신만 계속해서 동작시키는 것이 가장 좋을 것이기 때문이다. 이러한 문제를 해결하기 위해 강화학습에서는 각 Action의 기대 Reward를 추정(estimate)하고 추정치에 따라 Action을 결정하며 그 결과를 다시 학습하여 보다 정확하게 기대 Reward를 추정할 수 있도록 한다. 여기서 어떤 Action의 기대 Reward를 **Action Value**라고 한다.

## 1. Action Value Method

따라서 Multi-armed Bendit 문제를 해결하는 것은 각 Action들의 정확한 Action Value를 알아내는 것이다. 가장 간단하게 Action Value를 추정하는 방법은 일정 Time Step 동안 Action을 수행한 결과 얻은 Reward를 평균하는 것이다. 어떤 Action $$a$$에 대한 True Action Value를 $$q(a)$$라 하고, $$t$$시점에서 $$a$$를 $$N_t(a)$$번 경험했을 때 추정한 Action $$a$$의 Action Value를 $$Q_t(a)$$라고 표기한다면 다음과 같이 나타낼 수 있다.

$$
Q_t(a) = {R_1 + R_2 + ... + R_{N_t(a)} \over N_t(a)}
$$

여기서 $$N_t(a)$$가 $$N_t(a) \rightarrow \infty$$라고 한다면 $$Q_t(a)$$는 $$q(a)$$에 수렴하게 되는데, 이를 **Sample-average method**라고 한다. 그런데 Sample-average method는 하나의 Action 만을 반복하는 것을 가정하기 때문에 Action $$a$$에 대해서는 정확한 Action Value를 구했다 하더라도 다른 Action이 가지는 Action Value는 정확하게 계산하지 못한다. 이렇게 되면 $$a$$의 Value가 가장 높다 하더라도 정말 가장 높은 것인지 알 수 없다. **$$\epsilon$$-Greedy Method**는 이러한 문제를 가장 간단하게 해결하는 방법이다.

### 1) $$\epsilon$$-Greedy Method

 **$$\epsilon$$-Greedy Method**와 관련하여 용어를 먼저 정리하자면, Action Value $$Q_t(a)$$가 가장 큰 Action을 **Greedy Action**이라고 하고 그 이외의 Action들을 **Non-Greedy Action**이라고 한다. 그리고 Greedy Action만을 선택하는 것은 현재 추정하는 값을 이용한다는 점에서 **Exploitation**이라고 하고, Non-Greedy Action을 선택하는 것은 최선이 아닌 Action들의 Value를 확인한다는 점에서 **Exploration**이라고 한다.

위에서 말한 것과 같이 Action Value $$Q_t(a)$$는 무한 번 반복하지 않는 이상 정확한 값이 아니라 어디까지나 추정치일 뿐이다. 따라서 현재 Greedy Action보다 더 좋은 True Action Value $$q(a)$$를 가지는 다른 Action이 존재할 가능성이 있다. 이러한 불확실성을 해소하기 위해서는 Exploration, 즉 Non-Greedy Action들도 경험하여 이들에 대한 추정치도 일정 수준 이상으로 정확히 할 필요가 없다.

#### Greedy Action or Non-Greedy Action

따라서 최적의 Action을 선택하기 위해서는 Exploitation을 통해 현재 Greedy Action이 가지는 $$Q(a)$$를 정확히 평가하는 것도, Non-Greedy Action의 $$Q(a)$$을 정확히 평가하는 것도 중요하다. **$$\epsilon$$-Greedy Method**는 이들 간의 비율을 조절하는 방법으로 Action을 선택할 때 $$0 \leq \epsilon \leq 0$$의 확률로 임의의 Action을 선택(Exploration)하고 $$1 - \epsilon$$의 확률로 Greedy Action을 선택(Exploitation)하는 것을 말한다. 참고로 Sutton에 따르면 $$\epsilon$$-Greedy Method를 적용하게 되면 Greedy Method와 비교해 볼 때 수렴 속도는 조금 느리나 높은 수준에서 수렴하게 된다고 한다.

**$$\epsilon$$-Greedy Method**에서 $$\epsilon$$을 얼마나 크게 할 지 결정하는 것은 학습의 방향에 직접적인 영향을 미칠 정도로 매우 중요하다. 당연한 이야기지만 결론부터 말하자면 Task에 따라 적절한 $$\epsilon$$이 다르기 때문에 정해진 답이 없다. $$\epsilon$$의 크기와 관련된 Task의 특성으로는 어떤 Action에 따른 Reward의 Variance 이다. 만약 동일한 Action에 대한 Reward의 Variance가 0이라면 Exploration을 적게 해도 되지만 Reward의 Variance가 크면 클수록 Exploration을 많이 해야 Action Value의 정확도를 높일 수 있다.

### 2) Incremental Implementation

Sample Average Method가 가지는 문제점 중 하나는 $$Q_t(a)$$를 계산하기 위해서는 $$t$$개의 Reward를 모두 가지고 있어야 한다는 점이다. 즉 모든 Action $$a = 1, 2, ..., n$$의 Action Value

$$
Q_t(a) = {R_1 + R_2 + ... + R_{N_t(a)} \over N_t(a)}
$$

를 구하기 위해서는 지금까지의 모든 $$a$$에 대한 Reward가 메모리 상에 저장되어 있어야 한다. 이러한 문제를 해결하기 위해 $$Q_t(a)$$를 점진적으로 업데이트하며 $$Q_t(a)$$만을 저장하는 방법을 생각해 볼 수 있는데, 이것이 바로 **Incremental Implementation**이다. 아래 수식에서도 확인할 수 있듯이 Incremental Implementation는 Sample Average Method와 완전히 동일하다.

$$
\eqalign{
Q_{k+1} &= {1 \over k} \Sigma_i^{k}R_i \\
&= {1 \over k}(R_k + \Sigma_i^{k-1}R_i) \\
&= {1 \over k}(R_k + (k - 1)Q_k + Q_k - Q_k) \\
&= {1 \over k}(R_k + kQ_k - Q_k) \\
&= Q_k + {1 \over k} (R_k - Q_k) \\
}
$$

위의 식에서 $$Q_{k}$$는 기존 추정치를 의미하며 $$Q_{k+1}$$는 $$a$$에 대한 새로운 추정치가 된다. 그리고 $$(R_k - Q_k)$$는 $$t$$시점에 확인된 Reward와 추정치 간의 차이를 의미한다. 이에 따르면 현재 받은 Reward와 추정 Reward 간의 차이에 따라 새로운 추정치가 결정되는 것으로 볼 수 있다.

### 3) Recency-Weighted Average

한 가지 더 생각해 볼 만한 문제점은 어떤 Action의 True Action Value가 변화하는 경우, 즉 각 슬롯 머신의 당첨될 확률과 당첨금이 시간에 따라 달라지는 상황에는 어떻게 대처할 것인가이다. 이러한 문제를 해결하기 위해 최근에 경험한 Action의 Reward에 조금 더 높은 가중치를 부여하는 방법을 생각해 볼 수 있다. 위의 Incremental Implementation 식 $$Q_{k+1} = Q_k + {1 \over k} (R_k - Q_k)$$에서 $${1 \over k}$$를 대신하여 $$0 < \alpha \leq 1$$를 사용하면 쉽게 구현이 가능하다. 이때 $$\alpha$$를 Step Size라고 한다.

### 4) Optimistic Initial Value

위의 식들을 잘 보게 되면 $$Q_0$$, 즉 초기에 주어진 Action Value가 마지막 Action Value를 결정하는 데에도 영향을 미친다는 것을 알 수 있다. 무수히 많이 반복하게 된다면 이에 따른 bias를 없앨 수 있겠지만 현실적으로 불가능하므로 초기 값을 어떻게 설정할 것인가 또한 중요한 문제가 된다.

이와 관련하여 한 가지 재미있는 것은 Initial Value를 높게 설정하면 할수록 Exploration을 많이 하게 된다는 것이다. 예를 들어 모든 Action의 $$q(a)$$가 0인데 Initial Value는 모두 +5라고 하자. 이 경우 Action에 대한 Reward는 대개 5보다 작은 값이 될 것이다. 따라서 경험한 Action들은 모두 다음 $$Q_1$$이 $$Q_0$$보다 작을 확률이 높게 되고, Greedy Action Method에 따라 결과적으로 모든 Action들을 돌아가며 선택하게 된다는 것이다. 이와 같이 True Action Value보다 높은(Optimistic) Initial Value를 설정하여 Exploration을 높이는 방법을 **Optimistic Initial Value**라고 한다.

### 5) Upper Confidence-Bound Action Selection

Exploration이 필요한 이유가 Estimated Action Value의 불확실성 때문이라면 이러한 불확실성의 크기에 따라 Exploration의 수준을 조절하는 방법도 생각해 볼 수 있다. 이를 구현하기 위해서는 불확실성을 평가할 방법이 필요한데, **Upper Confidence-Bound Action Selection(UCB)** 에서는 Action $$a$$를 선택한 횟수에 따라 결정한다. 구체적으로는 아래와 같이 $$t$$시점에서의 Action $$A_t$$를 결정하게 된다.

$$
A_t = \arg_a \max [Q_t(a) + c \root \of {\ln t \over N_t(a)}]
$$

$$t$$ 번의 기회 중에서 Action $$a$$를 많이 선택하면 할수록 작아지는 값 $$\root \of {\ln t \over N_t(a)}$$이 Action $$a$$의 Value가 가지는 불확실성의 정도를 의미한다. 이 불확실성의 수준이 높으면 $$a$$를 선택할 가능성이 높아진다. Sutton의 책에 따르면 UCB를 사용할 때 $$\epsilon$$-Greedy Method에 비해 보다 높은 성능을 보일 수도 있다고 한다.

## 2. Gradient Bandit Algorithm

Action Value를 추정하는 방법 외에도 다른 방법들이 있는데 그 중 대표적인 것이 각 Action에 대한 선호(preference)를 기준으로 Action을 선택하는 방법이다. 선호는 상대적인 것이므로 어떤 Action $$a$$에 대한 $$H(a)$$가 주어져있을 때 해당 Action이 선택될 확률은 다음과 같이 모든 Action의 선호에 대한 softmax 값으로 정해진다.

$$
Pr(A_t = a) = {e^{H_t(a)} \over {\Sigma_{b=1}^n}e^{H_t(b)}} = \pi_t(a)
$$

그렇다면 $$H_t(a)$$는 어떻게 구할 수 있을까. Gradient Bandit Algorithm에서 Gradient라는 표현에 걸맞게 다음과 같이 Expected Reward $$E[R_t]$$를 극대화하는 방향으로 Gradient Ascent 방법을 통해 업데이트된다.

$$
H_{t+1}(a) = H_t(a) + \alpha{\partial E[R_t] \over \partial H_t(a)}
$$

이때 Expected Reward는 다음과 같이 각 Action이 가지는 Action Value의 가중평균으로 정의된다.

$$E[R_t] = \Sigma_b \pi_t(b)q(b)$$

Gradient를 구하는 방법은 다음과 같다.

$$
\eqalign{
{\partial E[R_t] \over \partial H_t(a)} &= {\partial \over \partial H_t(a)}[\Sigma_b \pi_t(b)q(b)] \\
&= \Sigma_b q(b) {\partial \pi_t (b) \over \partial H_t(a)} \\
&= \Sigma_b (q(b)-X_t) {\partial \pi_t (b) \over \partial H_t(a)}
}
$$

여기서 $$\Sigma_b {\partial \pi_t (b) \over \partial H_t(a)} = 0$$이므로 임의의 scalar 값 $$X_t$$를 빼주는 것이 가능하다. 위의 식에서 $$X_t$$를 전체 Action에 대한 Expected Reward $$\tilde R$$로 대체하면

$$
\eqalign{
&= \Sigma_b \pi_t(b)(q(b)-R_t) {\partial \pi_t (b) \over \partial H_t(a)} / \pi_t (b) \\
&= E[(q(A_t) - R_t) {\partial \pi_t (A_t) \over \partial H_t(a)} / \pi_t (A_t)] \\
&= E[(R_t - \tilde R_t) {\partial \pi_t (A_t) \over \partial H_t(a)} / \pi_t (A_t)] \\
}
$$

와 같이 바꿀 수 있다. 이때 $$\Sigma_b \pi_t(b)$$를 $$b$$에 대한 기대값으로 바꾸면서 임의의 어떤 Action $$A_t$$에 대한 식으로 바꾸었다. $$I_{a=b}$$를 $$a=b$$가 맞으면 1, 그렇지 않으면 0인 조건식이라고 한다면 $${\partial \pi_t(b) \over \partial H_t (a)} = \pi_t(b) (I_{a=b} - \pi_t(a))$$로 정리할 수 있다.

$$
\eqalign{
&= E[(R_t - \tilde R_t) \pi_t(A_t) (I_{a=b} - \pi_t(a)) / \pi_t (A_t)] \\
&= E[(R_t - \tilde R_t)(I_{a=b} - \pi_t(a))] \\
}
$$

최종적으로 이를 $$H_{t+1}(a) = H_t(a) + \alpha{\partial E[R_t] \over \partial H_t(a)}$$에 적용하면 다음과 같다.

$$
H_{t+1}(a) = H_t(a) + \alpha (R_t - \tilde R_t)(I_{a=b} - \pi_t(a))
$$

식을 풀어써보면 선택한 Action $$a$$에 대해서는

$$
H_{t+1}(a) = H_t(a) + \alpha (R_t - \tilde R_t)(1 - \pi_t(a))
$$

에 따라 업데이트되고 이외의 다른 Action들에 대해서는

$$
H_{t+1}(a) = H_t(a) - \alpha (R_t - \tilde R_t) \pi_t(a)
$$

에 따라 업데이트된다. 만약 $$(R_t - \tilde R_t) > 0$$, 즉 Action $$A_t$$에 따른 Reward가 전체 Action의 Expected Reward $$\tilde R$$보다 크다면 다른 Action들을 선택했을 때보다 더 큰 Reward를 기대할 수 있다는 것을 의미하므로 $$H(a)$$는 보다 커지게 되고, 이에 맞춰 다른 Action들에 대한 선호는 줄어들게 된다. 이와 같이 Gradient에 따라 업데이트하며 어떤 Action을 할 지 정하는 방법을 Gradient Bandit Algorithm이라고 한다.
