---
layout: post
title: TD3) Addressing Function Approximation Error in Actor-Critic Methods
category_num: 2
---

# 논문 제목 : Addressing Function Approximation Error in Actor-Critic Methods

- Scott Fujimoto 등
- 2018
- [논문 링크](<https://arxiv.org/abs/1802.09477>)
- 2019.11.07 정리

## 세 줄 요약

- 2018년 기준 SOTA의 성능을 보이는 **TD3** 알고리즘을 제시하고 있다. 본 논문에서는 TD3가 SAC보다 높은 성능을 보이지만, SAC 논문을 보면 TD3가 더 낮은 성능을 보인다. 2019년의 다른 논문들을 보면 두 알고리즘을 모두 SOTA로 보는 경향이 있다.
- DDPG에는 Actor-Critic의 특성상 발생하는 Overestimation bias 문제, high variance of target value 문제, susception to local maximum of target value 문제 등이 있으며, 이로 인해 안정적인 학습이 어렵다.
- TD3에는 두 개의 Critic을 구현하고 그 중 낮은 target value를 선택하고(Twin), 민감한 policy를 Critic에 비해 덜 업데이트하며(Delayed), 그리고 target value를 계산하는 데에 사용되는 action에 노이즈를 추가하여(noise to target policy) 이러한 문제들을 해결한다.

## 내용 정리

Function approximation error와 그것의 영향으로 추정치에 bias가 생기고 높은 variance가 나타나는 문제는 강화학습에서 오랫동안 연구되어온 주제였다. TD3 알고리즘은 그 중에서도 estimation error라는 이름으로 다루어지는 두 가지, overestimation bias 문제와 high variance 문제를 해결하는 것에 초점을 맞추고 있다.

Bias는 실제값과 예측값 간의 차이를 말하고, Variance는 예측값들이 흩어진 정도를 의미한다. 두 개념에서 짚고 넘어갈 점이 있다면 bias는 실제값(true value)과 예측값 간의 관계와 관련된 개념이지만 variance는 예측값들 간의 관계와 관련있다는 점이다. 이 점을 생각하면 overestimation bias 문제와 high variance 문제의 차이가 더 쉽게 와닿을 것이다.

### Overestimation bias

DQN을 비롯한 Descrete action을 학습하는 Q-learning 알고리즘에서는 overestimation bias 문제가 발생한다. 이는 Q target을 구할 때 여러 action 중 q value가 가장 큰 action을 선택(greedy target)하기 때문이다.

$$y = E[r + \gamma \max_{a'}Q(s', a') | s, a ]$$

즉, 위와같은 DQN 알고리즘 수식의 $$\max_{a'} Q(s',a')$$를 구하는 과정에서 Q value에 bias가 끼게 되면 y를 실제보다 과도하게 높은 값으로 구하게 되는 문제가 발생한다(Trun&Schwartz, 1993). 한마디로 error가 더해진 값이 그렇지 않은 값보다 커지는 경향이 있다는 것인데, 수학적으로는 다음과 같이 표현된다.

$$E_{\epsilon}[\max_{a'}Q(s',a') + \epsilon] \geqq \max_{a'}Q(s',a')$$

Overestimation bias 문제는 current Q value와 Q target value 간의 차이를 통해 loss를 구하는 Q-learning의 특성상 Q target value에 생긴 bias는 단일 업데이트의 문제로 끝나는 것이 아니라 누적(accumulation)된다. 결과적으로 누적된 overestimation error에 의해 안정적인 수렴이 어려워진다.

#### Overestimation bias in Actor Critic

Actor Critic 계열의 모델에서도 Critic 부분이 Q learning을 기초로 학습하기 때문에 overestimation bias에서 자유롭지 못하다. 다만 Q 네트워크를 이용해 action을 바로 결정하지 않고, Q 네트워크는 critic이라는 보조적인 장치로 기능하고, policy 자체는 gradient descent 방식에 따라 업데이트 된다는 점에서 Q 네트워크에서 발생하는 over estimation error가 실제 policy에 미치는 영향은 다소 불명확한 것이 사실이다. 논문에서는 실험을 통해 DPG 모델에서도 overestimation bias 문제가 발생할 수 있음을 보이고 있다. 그리고 부정확한 value estimation은 policy update에 문제를 일으키게 된다고 주장한다.

##### 1. DDQN

DDPG에서도 Q 네트워크(critic)에 overestimation bias가 발생한다면 discrete action Q learning에서 사용한 방법으로 해결할 수 있지 않을까 하는 질문이 가장 먼저 떠오른다.

$$y = r + \gamma Q_{\theta'}(s', \pi_\varnothing(s'))$$

하지만 DDQN([Van Hasselt et al, 2016](<http://localhost:4000/paper-review/reinforcement-learning/q-learning/2019/09/17/DDQN-Deep-Reinforcement-Learning-with-Double-Q-learning/>))과 같은 방법을 바로 적용하는 것은 DDPG를 비롯한 Actor Critic 계열의 모델에서는 충분히 효과적이지 않다(ineffective)고 한다. 이는 Actor Critic의 경우 policy가 상대적으로 천천히 변화하고, 이로 인해 current Q function과 target Q function 간의 차이가 크지 않아 단순히 두 개의 네트워크를 사용하는 것만으로는 이점이 적기 때문이다.

##### 2. Double Q-learning

clipped Double Q-learning for Actor-Critic은 논문에서 value estimation을 줄이기 위한 방법으로 제시한 방법이다. DDQN이 아니라 그것의 이론적 바탕이 된 Double Q-learning의 아이디이를 곧바로 적용한다. 즉 아래 수식과 같이 Actor와 Critic을 각각 두 개씩 만들고 Critic1은 Actor1을 기준으로, Critic2는 Actor2를 기준으로 업데이트하는 방법이다.

$$y_1 = r + \gamma Q_{\theta_2'}(s', \pi_{
\varnothing_1}(s'))$$

$$y_2 = r + \gamma Q_{\theta_1'}(s', \pi_{
\varnothing_2}(s'))$$

하지만 위와 같은 방법 또한 Actor-Critic 구조에서는 그렇게 효과적이지 못하다고 한다. 우선 단순히 두 개로 나누어 업데이트 할 경우 overestimation bias를 완벽하게 제거했다고 할 수 없다. 두 개의 네트워크를 사용한다는 것뿐이지 양쪽 모두에서 생기는 bias는 제거하지 못하기 때문이다. 게다가 DQN 계열에서는 target value를 구할 때에도 Q function을 사용하지만, Actor-Critic의 Critic에서는 Q function에 따라 update된 policy를 사용하기 때문에 단순히 Q function을 두 개 두는 것만으로는 그 효과가 제한적이다.

##### 3. clipped Double Q-learning for Actor-Critic

논문에서 제시하는 **clipped Double Q-learning**는 Double이라는 표현에서 알 수 있듯이 두 개의 Critic network를 두는 것은 위의 Double Q-learning과 동일하다. 다만 업데이트 방식에 있어 위의 방법과 차이가 있다. 즉 각각 서로 다른 Actor를 업데이트하는 것이 아니라 하나의 Actor만 업데이트하고 그 때의 Q value로는 두 개의 네트워크 아웃풋 중 작은 것을 사용한다.

$$y_1 = r + \gamma \min_{i=1,2}Q_{\theta_i'}(s', \pi_{\varnothing_1} (s'))$$

이는 만약 $$\max_{𝗮'} Q(s', a')$$ 값에 bias가 존재한다면 $$\max_{𝗮'} Q(s', a')$$ 값 자체를 true value의 상한선으로 보자는 아이디어에서 시작한다. 즉, 두 개의 estimation 모두 bias가 존재한다고 가정하고, bias가 덜 존재하는 값을 true value로 설정하여, 약간이라도 bias를 줄일 수 있다는 것이다.

### variance of target value estimation

estimator의 variance가 높은 것은 위에서 다룬 overestimation bias를 높이는 원인이 될 수 있다는 점에서도 문제가 되지만 variance가 높다는 상황 자체만으로도 policy의 업데이트에 부정적인 영향을 미칠 수 있다. 보다 구체적으로 estimator의 variance가 높으면 policy 업데이트에 noisy gradient가 사용되기 때문에 policy의 성능이 떨어지는 원인이 된다.

사실 function approximation의 특성상 variance가 발생하지 않을 수 없다. 오히려 variance가 0이라면 그 자체로 다른 문제를 내포하고 있을 가능성이 높다(variance-bias trade off). 즉 강화학습에서 개별 업데이트에 있어 어느 정도 작은 오차가 있을 것이라 기대하는 것은 이상한 것이 아니라 오히려 정상적인 것이다. 하지만 TD 방법을 이용하면 이러한 오차가 누적되기 때문에 문제가 되는 것이다.

이와 관련해 TD3에서는 policy를 업데이트하는 횟수(step) 당 estimation의 variance를 줄이기 위해 Actor를 업데이트하는 횟수를 줄이는 방법을 사용한다. 이를 두고 Delayed라는 표현을 쓰고 있다. 여기에 덧붙여 TD-error가 작게 유지되기를 바라므로, target 네트워크를 업데이트하는데 있어 가중 평균을 사용하는 soft update를 도입하고 있다.

soft update : $$\theta' \leftarrow \tau\theta + (1 - \tau)\theta'$$

#### Delayed의 의미 - target network와 policy의 관계

Actor-Critic에서 policy가 좋지 않으면 overestimation bias로 인해 target estimator가 제대로 학습되지 못한다. 반대로 target estimator가 정확하지 못하면 policy 또한 좋은 방향으로 학습되지 않는다. 즉 Actor와 Critic의 성능은 상호보완적인 관계를 가진다. target estimator의 성능이 낮은 상태에서 policy가 학습되는 것이 문제의 원인이라고 본다면, target estimator를 업데이트하는 횟수보다 더 낮은 횟수로 policy를 업데이트 하는 방법을 해결책으로 제시할 수 있다. Critic을 d번 업데이트 할 때 Actor를 1번 업데이트 하는 식이며, 이러한 점에서 delayed라고 한다.

### target policy smoothing regularization

Deterministic policy의 문제점 중 하나는 value estimation의 narrow peak에 오버피팅 될 수 있다는 점이다. 즉 deterministic policy가 overestimation error에 더 민감하다(susceptable to inaccuracies). 이러한 문제를 해결하기 위해 SARSA에서 regularization strategy를 위해 사용된 방법(Sutton&Barto 1998)과 비슷한 방법을 시도한다. 간단히 말해 target value를 구할 때 사용되는 a'에 NOISE를 더하는 것이다. 이를 통해 동일한 state s와 action a에 있어 target value가 부드러워(smoothing)지는 효과가 생기게 되며, narrow peak 문제에 덜 민감해진다고 한다. Noise는 정규분포를 가정하며, original action에서 크게 벗어나지 않도록 clipping이 이뤄진다.

$$y = r + \gamma Q_{\theta'}(s', \pi_{\varnothing'}(s') + \epsilon)$$

$$\epsilon - clip(N(0, \sigma), -c, c)$$

이 방법은 비슷한 action은 비슷한 value를 가진다는 점을 가정한다.

### TD3(Twin Delayed Deep Deterministic Policy Gradient)

TD3는 위에서 언급한 것과 같이 DDPG에서 빈번히 발생한 문제들을 해결하기 위해 알고리즘을 개선한 것이라고 할 수 있다. 즉 기본적으로는 이름에서도 알 수 있듯이 DDPG의 구조를 가지고 있다. 구체적으로 TD3에는 다음과 같이 세 가지 세부 알고리즘이 구현되어 있다.

  1. **Twin(clipped)**
    - DDPG와 마찬가지로 두 개의 Critic을 가지고 있지만, target value y 를 구할 때에는 둘 중 작은 target value 를 사용한다.
    - 이를 통해 overestimation bias를 낮춘다.
  2. **Delayed**
    - Critic이 d번 업데이트 될 때, Policy는 1번만 업데이트 한다.
    - 이를 통해 보다 안정적인 policy 업데이트가 가능해진다.
  3. **noise to target policy**
    - target value를 구할 때 사용되는 action에 노이즈를 더해 사용한다.
    - 이를 통해 target value의 narrow peak(또는 local minimum)에 덜 민감해진다.

#### Algorithm of TD3

```
Initialize critic networks Qθ₁ , Qθ₂ , and actor network πφ with random parameters θ₁, θ₂, φ
Initialize target networks θ₁′ ← θ₁, θ₂′ ← θ₂, φ′ ← φ Initialize replay buffer B
for t=1 to T do
    Select action with exploration noise a ∼ πφ(s) + ε, ε ∼ N (0, σ) and observe reward r and new state s′ Store transition tuple (s, a, r, s′ ) in B
    Sample mini-batch of N transitions (s, a, r, s′ ) from B
        a ̃←πφ′(s′)+ε, ε∼clip(N(0,σ ̃),−c,c)
        y ← r + γ min ᵢ₌₁,₂ Qθᵢ′ (s′, a ̃)
    Update critics θᵢ ← argminθᵢ N⁻¹ Σ(y−Qθᵢ (s, a))²
    if t mod d then
        Update φ by the deterministic policy gradient:
            ∇φJ(φ) = N⁻¹ Σ ∇aQθ₁ (s, a)|a=πφ(s)∇φπφ(s)
        Update target networks:
            θᵢ′ ←τθᵢ+(1−τ)θᵢ′
            φ′ ←τφ+(1−τ)φ′
        end if
end for
```

- 2번째 줄에서 선언하는 것과 같이 target network 를 별도로 두는 방법은 DDPG에서도 사용되는 방법이다.
- 7번째 줄에서 target value y를 구하는 데에 사용되는 action πφ′(s′) 에 노이즈 ε를 더하고 있다(noise to target policy). 이 때 action a~를 구할 때 사용되는 policy는 2번째 줄에서 선언한 target network이다.
- 8번째 줄에서 더 작은 Q value를 가지는 critic 네트워크를 이용해 y값을 구하고 있다(clipped).
- 10번째 줄에서 critic을 업데이트하지만 9번째 줄에서 policy 업데이트는 d번에 1회로 제한하고 있다(delayed).
- 14, 15 번째 줄에서 이뤄지는 soft update는 DDPG에서도 사용하던 방법이다. 이 방법은 실제 Actor, Critic과 target Actor, Critic 간에 차이가 너무 크지 않도록 하는 동시에 target network의 안정적인 변화를 가능하게 한다.
