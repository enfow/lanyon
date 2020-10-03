---
layout: post
title: "Mujoco Env: Reacher & Hopper"
keyword: '[RL]'
category_num : 5
---

# Mujoco Env: Reacher & Hopper

- [Mujoco 홈페이지](<http://www.mujoco.org/>)
- [Gym Mujoco Envs](<https://gym.openai.com/envs/#mujoco>)
- update at 2020.10.02

## Reacher-v2

**Reacher-v2**는 아래 이미지와 같이 두 개의 관절로 이뤄진 로봇팔을 조작하는 환경으로, 끝 부분(Fingertip)을 임의로 생성되는 목표지점에 도달시키는 것을 목표로 한다. 

이미지 - gym_reacher_v2

Python Package gym과 mujoco-py가 설치된 환경에서 다음과 같이 생성할 수 있다.

```python
import gym

env_name = "Reacher-v2"
env = gym.make(env_name)
```

Reacher-v2는 11차원의 State를, 2차원의 Action을 가진다. Reward는 일반적인 다른 강화학습 문제와 마찬가지로 Scalar 값으로 주어진다.

### State of Reacher-v2

보다 구체적으로 Reacher-v2의 State는 다음과 같이 구성된다.

```python
# gym/envs/mujoco/reacher.py

def _get_obs(self):
    theta = self.sim.data.qpos.flat[:2]
    return np.concatenate([
        np.cos(theta),
        np.sin(theta),
        self.sim.data.qpos.flat[2:],
        self.sim.data.qvel.flat[:2],
        self.get_body_com("fingertip") - self.get_body_com("target")
    ])
```

각각이 가지는 의미는 다음과 같다.

- sim.data.qpos.flat[:2] : 로봇팔 첫 번째 축과 두 번째 축의 각도
- sim.data.qpos.flat[2:] : Target의 x,y 좌표
- sim.qvel.qpos.flat[:2] : x,y 방향에 대한 Fingertip의 속도

임의의 값을 찍어보면 다음과 같다.

|obj|value|dim(as state)|
|:---:|:----:|:---:|
|theta| [-0.02911098, -0.00565784] | 2(0) |
|np.cos(theta)| [-0.02911098, -0.00565784] | 2(2) |
|np.sin(theta)| [0.99957631, 0.99998399] | 2(2) |
|self.sim.data.qpos.flat| [-0.02911098, -0.00565784.  0.1.        -0.1       ] | 4(2) |
|self.sim.qvel.qpos.flat| [-2.9014378,  -0.56390888, 0.,          0.        ] | 4(2) |
|("fingertip")-("target")| [0.10989124, 0.09326832, 0.,        ] | 3(3) |

### Action of Reacher-v2

Reacher-v2의 Action은 2차원으로, 각각 첫 번째 관절과 두 번째 관절의 각도를 의미한다. 각각에 대해 최대값과 최소값은 모두 1, -1이며 다음과 같이 확인할 수 있다.

```python

env.action_space.high   #[1. 1.]
env.action_space.low    #[-1. -1.]
```

### Reward of Reacher-v2

Reward를 계산하는 함수는 다음과 같다.

```python
# gym/envs/mujoco/reacher.py

# def step(self, a):
    vec = self.get_body_com("fingertip")-self.get_body_com("target")
    reward_dist = - np.linalg.norm(vec) # Vector의 Norm
    reward_ctrl = - np.square(a).sum()  # Action의 제곱 합
    reward = reward_dist + reward_ctrl
```

Reacher-v2의 Reward는 **reward_dist**와 **reward_ctrl**의 합으로 계산되며, 두 가지 모두 음수이므로 Reacher-v2의 Reward는 항상 음수가 된다.

- **reward_dist**: Distance의 약자로, fingertip과 target 간의 유클리디안 거리에 음수를 취한 것이다.
- **reward_ctrl**: Control의 약자로, Step에 주어진 Action의 크기에 음수를 취한 것이다.

각각의 값을 한 번 확인해보면 다음과 같다.

|obj|value|
|:---:|:----:|
|a(action) |[-0.87656677  0.7254853 ] |
|vec |[0. 0. 0.] |
|reward_dist |-0.0 |
|reward_ctrl |-1.2946982 |
|reward |-1.2946982383728027 |

### Reset of Reacher-v2

Reacher-v2를 reset 했을 때 호출되는 함수는 다음과 같다.

```python
def reset_model(self):
    qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
    while True:
        self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        if np.linalg.norm(self.goal) < 0.2:
            break
    qpos[-2:] = self.goal
    qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
    qvel[-2:] = 0
    self.set_state(qpos, qvel)
    return self._get_obs()
```

---

## Hopper-v3

**Hopper-v3**는 아래와 같이 외발이 중심을 잡아가며 앞으로 나아가도록 하는 환경이다. 따라서 넘어지지 않고 앞으로 멀리가면 갈수록 높은 Reward를 받는다.

이미지 - gym_hopper_v3

<img src="{{site.image_url}}/development/gym_hopper_v3.png" style="width:22em; display: block; margin: 0px auto;">

Hopper-v3의 State는 11차원, Action은 3차원이다.

### State of Hopper-v3

State를 제공하는 함수는 다음과 같다.

```python
# gym/envs/mujoco/hopper_v3.py

def _get_obs(self):
    position = self.sim.data.qpos.flat.copy()
    velocity = np.clip(
        self.sim.data.qvel.flat.copy(), -10, 10)

    if self._exclude_current_positions_from_observation:
        position = position[1:]

    observation = np.concatenate((position, velocity)).ravel()
    return observation
```

11차원은 크게 5차원의 위치 정보와 6차원의 속도 정보로 나누어 구해진다는 것을 쉽게 확인할 수 있다.

|obj|value|
|---|---|
| position | [ 1.24970027e+00  1.32572683e-03  1.04814497e-03  1.03141550e-03 -1.41838609e-03] |
| velocity | [ 0.01116277 -0.07309075  0.15096306  0.12359146  0.1217296  -0.34526038] |

### Action of Hopper-v3

Hopper-v3의 Action의 최대값과 최소값은 세 요소 모두 1, -1이다.

```python
env.action_space.high   #[1. 1. 1.]
env.action_space.low    #[-1. -1. -1.]
```

### Reward of Hopper-v3

```python
# gym/envs/mujoco/hopper_v3.py

def step(self, action):
    x_position_before = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    x_position_after = self.sim.data.qpos[0]
    x_velocity = ((x_position_after - x_position_before)
                    / self.dt)

    ctrl_cost = self.control_cost(action)

    forward_reward = self._forward_reward_weight * x_velocity
    healthy_reward = self.healthy_reward

    rewards = forward_reward + healthy_reward
    costs = ctrl_cost

    observation = self._get_obs()
    reward = rewards - costs

def control_cost(self, action):
    control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
    return control_cost
```

Hopper-v3의 Reward는 다음 식에 의해 구해진다.

- reward = forward_reward + healthy_reward - ctrl_cost

각각에 대해 살펴보면 다음과 같다.

#### forward_reward

**forward_reward**는 얼마나 앞으로 전진했는가에 관한 Reward로, 당연히 멀리가면 갈수록 높은 Reward를 받는다.

```python
# gym/envs/mujoco/hopper_v3.py
forward_reward = self._forward_reward_weight * x_velocity
```

위의 식을 보면 x축 방향으로의 속도가 높으면 높을수록 높은 Reward를 받는다는 것을 알 수 있다.

#### healthy_reward

**healthy_reward**는 생존 보너스라고 할 수 있다. 생존 시 1을, 사망 시 0을 받는다.

```python
# gym/envs/mujoco/hopper_v3.py
def healthy_reward(self):
    return float(
        self.is_healthy
        or self._terminate_when_unhealthy
    ) * self._healthy_reward
```

#### ctrl_cost

**ctrl_cost**는 Action의 크기가 크면 클수록 Panelty를 주어 불필요한 행동을 억제하는 역할을 수행한다. 계산 방식은 다음과 같다.

```python
# gym/envs/mujoco/hopper_v3.py
def control_cost(self, action):
    control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
    return control_cost
```

### values

Hopper-v3의 step에 사용되는 object들의 값을 예제로 찍어보면 다음과 같다.

|obj|value|
|:---:|:---:|
| action | [ 0.02057192 -0.14422667  0.13205169] |
| self.sim.data.qpos(before) | [0.   1.25 0.   0.   0.   0.  ] |
| self.sim.data.qpos(after) | [-5.71809482e-05  1.24967843e+00 -3.19840565e-04  4.19652123e-05] |
| x_position_before | 0.0 |
| x_position_after | -5.718094824325657e-05 |
| x_velocity | -0.007147618530407071 |
| ctrl_cost | 3.866218775510788e-05 |
| forward_reward | -0.007147618530407071 |
| healthy_reward | 1.0 |
| reward | 0.992852381469593 |
