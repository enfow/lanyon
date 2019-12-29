---
layout: post
title: TD3) Addressing Function Approximation Error in Actor-Critic Methods
---

# 논문 제목 : Addressing Function Approximation Error in Actor-Critic Methods

- Scott Fujimoto 등
- 2018
- <https://arxiv.org/abs/1802.09477>
- 2019.11.07 정리

## 세 줄 요약

- DDPG에는 Actor-Critic의 특성상 발생하는 Overestimation bias 문제, high variance of target value 문제, susception to local maximum of target value 문제 등이 있으며, 이로 인해 안정적인 학습이 어렵다.
- TD3에는 두 개의 Critic을 구현하고 그 중 낮은 target value를 선택하여 overestimation bias 문제를, 민감한 policy를 Critic에 비해 덜 업데이트하는 방법으로 high variance 문제를, 그리고 target value를 계산하는 데에 사용되는 action에 노이즈를 추가하여 local maximum 문제를 해결한다.
- 실험 결과 Hopper 기준으로 3000 점을 넘는 등 SOTA의 성능을 보였다.

## 내용 정리

### Overestimation bias

DQN을 비롯한 Q-learning 계열의 알고리즘에서는 overestimation bias 문제가 발생한다. 이는 Q target을 구할 때 여러 action 중 q value가 가장 큰 action을 선택(greedy target)하기 때문이다.

`y = E[r + 𝛾max𝗮' Q(s', a') | s, a]`

즉, 위와같은 DQN 알고리즘 수식의 `max𝗮' Q(s', a')`를 구하는 과정에서 overestimation bias가 Q value에 들어가게 되면 y를 실제보다 과도하게 높은 값으로 구하게 되거나, true maximum action이 아닌 다른 action의 값을 이용해 y를 구하는 문제가 발생한다(Trun&Schwartz, 1993). Overestimation bias 문제는 current Q value와 Q target value 간의 차이를 통해 loss를 구하는 Q-learning의 특성상 Q target value에 생긴 bias는 단일 업데이트의 문제로 끝나는 것이 아니라 누적(accumulation)된다. 결과적으로 누적된 overestimation error에 의해 안정적인 수렴이 어려워진다.

DQN에서 이러한 문제를 해결하기 위해 도입된 방법이 DDQN으로, Double이라는 표현에서 알 수 있듯 Q function을 두 개 만들어 current Q value를 구하는 파라미터와 target Q value를 구하는 파라미터를 서로 달리하는 방법이 있다.

#### Overestimation bias in Actor Critic

Actor Critic 계열의 모델에서도 Critic 부분이 Q learning을 기초로 학습하기 때문에 overestimation bias에서 자유롭지 못하다. actor가 아닌 critic의 문제이므로, deterministic policy를 쓰는 DDPG에서도 동일한 문제가 발생하게 된다. 하지만 DDQN과 같이 단순히 두 개의 Q function을 만드는 것은 DDPG를 비롯한 Actor Critic 계열의 모델에서는 효과적이지 않다(ineffective)고 한다. (이와 관련된 내용은 아래 clipped Double Q-learning for Actor-Critic 부분 참조) 이는 Actor Critic의 경우 policy가 상대적으로 천천히 변화하고, 이로 인해 current Q function과 target Q function 간의 차이가 크지 않아 단순히 두 개의 네트워크를 사용하는 것만으로는 이점이 적기 때문이다. 게다가 PG 계열의 경우 policy를 직접 학습하기 때문에 overestimation bias problem이 쉽게 드러나지 않게(less clear) 된다.

이와 관련하여 DDPG에서 overestimation bias가 발생함을 Hopper와 Walker2d 환경에서 실험을 통해 보이고 있다. 그리고 부정확한 value estimation은 policy update에 문제를 일으키게 된다고 주장한다.

#### Overestimation bias를 해결하는 방법

overestimation bias 문제를 해결하는 방법 중 하나는 앞서 언급한 DDQN과 같이 **current와 target을 구하는 estimator를 두 개 만드는 것**이다. 이를 통해 bias가 없는 target value 추정(unbiased value estimate)이 가능해진다.

다른 방법으로는 **직접적으로 target value estimation의 variance를 낮추는 것**에 집중하는 방법이다. 이와 관련하여 estimation의 오버피팅을 방지하는 방법, 직접적으로 variance를 낮추는 term을 추가하는 방법들이 연구되어졌다.

#### clipped Double Q-learning for Actor-Critic

clipped Double Q-learning for Actor-Critic은 논문에서 value estimation을 줄이기 위한 방법으로 제시한 방법으로, 기존의 DDQN에서 사용한 방법과 약간의 차이가 있다.

DDQN은 기초적인 DQN 모델에서 target을 구하는 네트워크를 별도로 두는 방법으로 overestimation bias를 줄이려고 한다. 이를 그대로 Actor-Critic 구조에 적용한다면, Actor와 Critic을 각각 두 개씩 만들고 Critic1은 Actor1을 기준으로, Critic2는 Actor2를 기준으로 업데이트하는 방법을 생각해 볼 수 있다.

```
  y1 = r+γQθ₁′(s′,πφ₁(s′))
  y2 = r+γQθ₂′(s′,πφ₂(s′))
```

하지만 DDQN과는 달리 위와 같은 방법은 Actor-Critic 구조에서는 그렇게 효과적이지 못하다고 한다. 우선 단순히 나누어 업데이트 할 경우 overestimation bias를 완벽하게 제거했다고 할 수 없다. 두 개의 네트워크를 사용한다는 것뿐이지 양쪽 모두에서 생기는 bias는 제거하지 못하기 때문이다. 게다가 DQN 계열에서는 target value를 구할 때에도 Q function을 사용하지만, Actor-Critic의 Critic에서는 Q function에 따라 update된 policy를 사용하기 때문에 단순히 Q function을 두 개 두는 것만으로는 그 효과가 제한적이다.

논문에서 제시하는 **clipped Double Q-learning**는 Double이라는 표현에서 알 수 있듯이 두 개의 Critic network를 두는 것은 DDQN과 동일하다. 다만 업데이트 방식에 있어 위의 방법과 차이가 있다. 즉 각각 서로 다른 Actor를 업데이트하는 것이 아니라 하나의 Actor만 업데이트하고 그 때의 Q value로는 두 개의 네트워크 아웃풋 중 작은 것을 사용한다.

`y = r + γ min Qθᵢ′ (s′, πφ₁ (s′))`

이러한 방법으노 `max𝗮' Q(s', a')` 값에 bias가 존재한다면 `max𝗮' Q(s', a')` 값 자체를 true value의 상한선으로 보자는 아이디어에서 시작한다. 즉, 두 개의 estimation 모두 bias가 존재한다고 가정하고, bias가 덜 존재하는 값을 true value로 설정하여, 약간이라도 bias를 줄일 수 있다는 것이다.

### variance of target value estimation

Estimation의 variance가 높으면 policy 업데이트가 잘 이뤄지지 않는다고 한다(Sutton&Barto, 1998). 일반적인 Temporal Differnece 방법을 사용하면 전체 에피소드의 Return이 아닌, 다음 step의 Reward만을 이용하여 target value를 추정하기 때문에 높은 varinace가 생긴다. 이와 관련해 variance를 줄이기 위한 방법들이 제시되어 왔는데, 첫 번째는 매 step마다 error(bias)를 줄이는 방법이고 다른 방법은 한 step이 아니라 여러 step을 진행하고 이를 통해 target을 추정하는 것이다. 하지만 후자와 같이 복수의 step에서 나온 Reward의 합으로 variance 문제를 해결하려고 한다면, step이 진행됨에 따라 overestimation bias이 누적되는 문제가 발생한다. 이러한 점에서 문제를 우회(circumvent)하는 방법에 불과하다고 보아야 할 것이다.

이와 관련해 TD3에서는 policy를 업데이트하는 횟수(step) 당 estimation의 variance를 줄이기 위해 Actor를 업데이트하는 횟수를 줄이는 방법을 사용한다.

#### Delayed의 의미 - target network와 policy의 관계

Actor-Critic에서 policy가 좋지 않으면 overestimation bias로 인해 target estimator가 제대로 학습되지 못한다. 반대로 target estimator가 정확하지 못하면 policy 또한 좋은 방향으로 학습되지 않는다. 즉 Actor와 Critic의 성능은 상호보완적인 관계를 가진다. target estimator의 성능이 낮은 상태에서 policy가 학습되는 것이 문제의 원인이라고 본다면, target estimator를 업데이트하는 횟수보다 더 낮은 횟수로 policy를 업데이트 하는 방법을 해결책으로 제시할 수 있다. Critic을 d번 업데이트 할 때 Actor를 1번 업데이트 하는 식이며, 이러한 점에서 delayed라고 한다.

### target policy smoothing regularization

Deterministic policy의 문제점 중 하나는 value estimation의 narrow peak에 오버피팅 될 수 있다는 점이다. 즉 deterministic policy가 overestimation error에 더 민감하다(susceptable to inaccuracies). 이러한 문제를 해결하기 위해 SARSA에서 regularization strategy를 위해 사용된 방법(Sutton&Barto 1998)과 비슷한 방법을 시도한다. 간단히 말해 target value를 구할 때 사용되는 a'에 NOISE를 더하는 것이다. 이를 통해 동일한 state s와 action a에 있어 target value가 부드러워(smoothing)지는 효과가 생기게 되며, narrow peak 문제에 덜 민감해진다고 한다. Noise는 정규분포를 가정하며, original action에서 크게 벗어나지 않도록 clipping이 이뤄진다.

`y=r+γQθ′(s′,πφ′(s′)+ε)` `ε ∼ clip(N (0, σ), −c, c)`

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
