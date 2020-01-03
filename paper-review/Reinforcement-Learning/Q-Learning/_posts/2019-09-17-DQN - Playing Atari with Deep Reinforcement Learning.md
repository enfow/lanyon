---
layout: post
title: DQN) Playing Atari with Deep Reinforcement Learning
category_num: 1
---

# 논문 제목 : Playing Atari with Deep Reinforcement Learning

- David Silver 등
- 2013
- 2019.09.17 내용 정리

## 세 줄 요약

- 강화학습에 Deep Neural Net을 적용하여 Current value function을 Optimal value function에 근사하도록 했다. 이를 **DQN(Deep Q-Networks)** 이라고 한다.
- state 정보로 raw video image를 원본대로 사용하지 않고 CNN을 통해 전처리하여 사용하면 Neural Net에 들어가는 input size가 동일해지는 등의 이점이 있다.
- **Experience Replay** 기법을 도입하여 RL에 Neural Net을 적용할 때 발생할 수 있는 문제를 비롯한 여러 문제들을 해결한다.

## 내용 정리

### 강화학습에 딥러닝을 적용하기 어려운 이유

1. 일반적인 딥러닝 기법들은 정해진 데이터셋(hand labelled training data)을 학습 대상으로 한다. 하지만 강화학습은 reward라는 scalar signal 학습 대상으로 하는데, sparse, noisy and delayed 한 특성 때문에 학습에 어려움이 있다.
2. 또한 다른 딥러닝 기법들은 각각의 data sample을 독립적으로 보지만 강화학습은 서로 연결된 transaction(state, action, reward) 들의 나열(sequence of highly correlated states)를 학습 대상으로 한다. 이러한 점에서 state 간의 correlation 이 문제가 되기도 한다.
3. 마지막으로 다른 딥러닝 기법들은 고정된 잠재 분포(fixed underlying distribution)를 가정하고 이에 근사하는 것을 목표로 한다. 하지만 강화학습은 알고리즘이 새로운 행동을 발견하면 가정하는 분포 자체가 달라진다. 이러한 점은 모델의 수렴을 어렵게 하는 요인이 된다.

### Bellman eqaution

모든 강화학습의 목표는 전체 episode에서 받을 것으로 기대되는 reward의 총합인 return G를 극대화하는 것이다. 이를 수학적으로 표현하기 위해 사용되는 것이 Bellman equation 이며, 논문 또한 여기서 시작하고 있다.

Bellman equation은 expectatation exquation과 optimality equation 두 가지가 있다.

- expectation equation : $$q_\pi(s, a) = R_{t+1} + \gamma q_\pi(s', a')$$
- optimality equation : $$ q^*(s, a) = \max_\pi q_\pi(s, a) $$

여기서 q 함수는 현재 policy $$\pi$$를 따라 state $$s$$ 에서 action $$a$$ 를 취했을 때 기대되는 return 의 기대값이다. expectation 식을 보면 state $$s$$ 에서 action $$a$$를 했을 때 환경으로부터 주어지는 보상 $$R_{t+1}$$ 과 다음 state $$s'$$과 policy $$\pi$$에 따라 결정된 다음 action $$a'$$의 q value 값의 합으로 되어 있다. 여기서 $$\gamma$$는 감가항이다. 

즉, expecation equation은 현재의 policy를 따를 때 받을 것으로 기대되는 return의 크기를 의미한다.

반면 optimality equation은 현재의 state $$s$$ 와 action $$a$$에서 기대되는 return을 극대화하는 policy 를 따를 때 받을 것으로 기대되는 return 값이다. 이 때 optimality equation에 근사하도록 expecation equation을 학습시키면 각 state에서 최적의 action을 선택할 것이라고 기대할 수 있다.

### DQN과 Bellman equation

DQN은 네트워크로 Bellman equation의 q value를 계산하게 되며, state를 네트워크의 입력으로 넣으면 선택 가능한 각 action의 q value가 모두 계산되어 나온다. 따라서 네트워크의 input dimension은 state size와 같고, output dimension은 action size와 같다. DQN은 Q 네트워크가 현재 state에서 가장 좋은 action에 대해 가장 높은 q value를 출력하도록 학습이 이뤄진다. 즉 policy는 매 state에서 q value가 가장 큰 action을 선택하는 것이며, policy의 질은 얼마나 정확한 q value를 출력하는가에 따라 결정된다.

Bellman optimality equation에 따라 action을 결정하게 되면 항상 return을 최대로 하는 action을 선택할 수 있게 된다. 이를 DQN에 적용하게 되면 Bellman optimality equation에 따라 계산된 q value를 target 값과, 현재 Q 네트워크를 통해 계산된 q value를 expectation 값 간의 차이(MSE)로 네트워크를 학습하게 된다.

- expectation q value

$$ q_i = Q(s, a; \theta_{i-1}) $$

- target q value

$$ y_i = r + \gamma \max_{a'}Q(s', a'; \theta_{i-1}) $$

- MSE loss

$$ L_i(\theta_i) = (y_i - Q(s, a; \theta_i))^2 $$

### DQN 알고리즘

```
Initialize replay memory D to capacity N
  Initialize action-value function Q with random weights
  for episode = 1, M do
    Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
    for t = 1, T do
      With probability ε select a random action at
      otherwise select at = maxa Q∗(φ(st), a; θ)
      Execute action at in emulator and observe reward rt and image xt+1
      Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
      Store transition (φt, at, rt, φt+1) in D
      Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
          if φj+1 is non-terminal : yj = rj + γ maxa′ Q(φj+1, a′; θ)
          else φj+1 is terminal : yj = rj
      Perform a gradient descent step on (yj − Q(φj , aj ; θ))2 according to equation 3
    end for
  end for
```

#### TD - Temporal difference

DQN에 가장 큰 영향을 미친 방법론은 **Temporal difference(TD)** 라고 할 수 있다. TD란 강화학습의 발전 과정에서 나온 두 가지 방법론 Monte Carlo와 Dynamic Programming 을 합친 개념으로, Monte Carlo 방식을 통해 샘플로 학습이 이뤄지고, Dynamic Programming처럼 전체 Episode가 끝나는 것을 기다리지 않고도 학습이 이뤄질 수 있도록 하는 방법이다.

Monte Carlo 를 수식으로 표현하면 다음과 같이 value function을 업데이트하게 된다.

$$ V(S_t) \leftarrow V(S_t) + \gamma(G_t - V(S_t)) $$

반면 TD는 다음과 같다.

$$ V(S_t) \leftarrow V(S_t) + \alpha ( R_t + \gamma V(S_{t+1}) - V(S_t) ) $$

Monte Carlo의 전체 episode return $$G_t$$가 $$R_t + \gamma V(S_{t+1})$$로 대체되었다. 즉, 매 step의 reward를 극대화하면 결과적으로 전체 episode의 return을 크게 할 수 있을 것이라는 아이디어이다. 전체 episode를 단위로 학습할 경우 variance가 크기 때문에 발산하지만 step 단위로 학습이 이뤄지는 TD 에서는 이러한 문제가 줄어들어 학습이 보다 쉬워진다. 이 과정에서 추정치를 이용하므로 overestimation bias 등이 생기기도 하나 학습이 이뤄지도록 하는 주된 방법이 된다(이를 보완하기 위해 DDQN이 나오게 되며, 나아가 Actor Critic 계열에서는 Twin critic 등의 방법들도 제안된다).

TD 에 따르면 학습되는 크기는 target v value **$$R_t + \gamma V(S_{t+1})$$**와 current v value **$$V(S_t)$$** 간의 차이가 된다. 이때 이 차이를 **TD error**라고 한다.

#### model-free

DQN은 model-free와 off-policy의 특성을 띤다. 정의를 그대로 옮겨 쓰자면, "it solves the reinforcement learning task directly using samples from the emulator ℰ, without explicitly constructing an estimate of ℰ." 즉, 환경에 대한 모델을 만들어 이용하는 것이 아니라 실제 환경을 경험하고 이를 통해 얻은 샘플을 대상으로 학습이 이뤄진다.

#### off-policy

off-policy란 TD의 학습에서 현재 state value와 다음 state value를 결정할 때 사용되는 policy가 다른 경우를 의미한다. DQN은 off-policy 의 특성을 가지는데, 현재 state-action value(q value)를 구할 때에는 epsilon greedy policy를 사용하지만 optimal(target) q value를 구할 때에는 greedy policy를 사용하기 때문에 서로 다르다.

#### replay buffer

DQN에서 network를 학습시키기 위해서는 current state s, curent action a, next reward r 그리고 next state s'이 필요하며 매 step 단위로 네 개의 조합이 발생하게 된다. 이를 transaction이라고 하는데, 이러한 transaction을 발생한 순서대로 학습시키게 되면 transaction 간의 상관관계로 인해 학습이 잘 되지 않는다고 한다. 이러한 문제를 해결하기 위해 논문에서는 replay buffer를 제시하고 있다.

replay buffer는 transaction을 저장하는 저장소로, 선입선출에 따라 일정 개수 이상의 transaction이 저장되면 맨 처음 저장된 transaction을 삭제하는 방식으로 동작한다. 그리고 학습의 대상이 되는 transaction은 이 buffer에서 랜덤 샘플링을 통해 결정하게 된다.

replay buffer의 이점으로 논문에서는 다음 세 가지를 제시하고 있다.

1.데이터를 반복적으로 학습에 이용하기 때문에 데이터를 보다 효율적으로 사용하게 된다.
2. random draw를 통해 선택된 sample의 집합을 학습대상으로 하므로 state간 상관관계 문제를 해결하는 데 도움이 된다.
3. divergence, local minimum 문제 등에 있어서 탁월한 성능을 보인다.

### 논문의 실험

- Atari game과 같이 raw video data에 대해서도 CNN을 이용하여 성공적인 강화학습이 가능하다는 것을 보여주고자 한다.
  - Atari 2600은 210X160 RGB 60Hz 의 RL testbed로 사용한다.
  - 각 게임은 일반적인 사람에게 어려운 수준이다.
  - 학습은 사람과 동일한 조건 하에서 이뤄진다.
    - 즉 "Video input, Reward, Terminal signal, Set of possible actions" 의 조건 하에서 학습이 이뤄진다.

- 논문 저자들은 목표로 다음 두 가지를 설정하고 있다.
    1. RGB image를 대상으로 학습하는 Deep Neural Net을 강화학습과 연결하는 것
    2. 학습과정에서 SGD를 사용하여 효율적으로 학습을 진행하는 것

- loss function과 구체적인 알고리즘은 각각 논문 3쪽, 5쪽 참고(위의 background 내용)
  - loss fucntion의 경우 현재 Q function을 통해 구해지는 값과 optimal Q 간의 차이의 제곱으로 표현된다.

- state, 즉 video image 는 CNN을 이용해 preprocessing 되어 모든 state가 동일한 크기를 가지도록 하고 있다.

- 논문의 실험에서 사용되는 기법 중 가장 특별한 것 중 하나는 Experience replay 이다.
