---
layout: post
title: Gym Documentation
---

# Gym Documentation

강화학습의 실험환경으로 많이 사용되는 GYM Environment의 문서를 번역한 것으로, [OpenAI](<http://gym.openai.com/docs/>)의 허락을 받고 올립니다. 원문의 내용을 최대한 유지하기 위해 최대한 문장 by 문장으로 번역했습니다.

---

**Gym은 강화학습 알고리즘을 개발하고 비교하는 데 사용되는 툴킷(toolkit)입니다.** agent의 구조에 대한 어떠한 가정도 하지 않으며, Tensorflow, Theano 같은 머신러닝 라이브러리와도 쉽게 호환됩니다.

Gym 라이브러리에는 강화학습 알고리즘을 작동시키는 데에 필요한 여러 environment들이 들어 있습니다. 이러한 environment들은 서로 인터페이스를 공유하고 있어 강화학습 알고리즘의 변경 없이도 여러 environment에서 실험이 가능합니다.

## 설치

#### 일반적인 방법 - pip

pip를 이용해 Gym 라이브러리를 설치하기 위해서는 Python 3.5 이상의 버전이 필요합니다.

`pip install gym`

#### git clone을 이용하는 방법

git clone을 통해서도 gym 라이브러리를 설치할 수 있습니다. gym 라이브러리를 수정하고 싶거나 새로운 환경을 추가하고 싶다면 git clone을 이용하는 방법이 유용합니다.

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

다만 이 방법을 이용할 경우 cmake, pip version 등 몇가지 의존성 문제가 발생할 수 있습니다.

## Gym의 구성요소

### Environment

environment를 이용하는 방법을 몇 가지 예시를 통해 알려주려 합니다. 아래의 코드는 *CartPole-v0* environment에서 1000 step만큼 진행하며, 매 step마다 이미지를 렌더링하도록 하고 있습니다. 렌더링된 이미지는 팝업창을 통해 확인할 수 있습니다.

```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```

보통은 cartpole이 스크린 밖으로 나가기 전에 시뮬레이션이 끝나게 됩니다. 이것과 관련해서는 아래에서 이어 설명하도록 하겠습니다.

만약 cartpole이 아닌 다른 환경에서의 action을 확인해보고 싶다면 gym.make()의 파라미터로 *CartPole-v0* 대신 *MountainCar-v0*, *MsPacman-v0* 또는 *Hopper-v1*를 전달하면 됩니다. 이때 각 environment는 서로 다른 의존성을 가지고 있습니다(Atari, Mujoco 등).

만약 의존성 문제가 해결되지 않는다면 도움이 되는 에러 메시지를 받을 수 있습니다. (만약 문제를 해결하는 데에 도움이 되지 않는다면 다음 [링크](https://github.com/openai/gym/issues)를 통해 알려주시기 바랍니다) 의존성 관리를 위한 다른 패키지의 설치는 매우 간단합니다. 다만 Mujoco의 경우에는 [라이센스](<https://www.roboti.us/license.html>)가 필요합니다.

### Observations

각 step마다 임의로 action을 선택하는 것(random action)보다 좋은 action을 선택하고 싶다면, 실제 action 들이 environment 내에서 어떻게 작동하는지 알 필요가 있습니다.

이때 environment의 step() 함수는 정확히 우리가 필요한 것을 반환합니다. 구체적으로 step() 함수는 다음 네 가지를 반환합니다.

- observation(*object*) : 입력 받은 action을 실행한 후 environment에 대한 관찰 결과를 반환합니다. 예를 들어 카메라로 찍은 픽셀 데이터, 로봇 관절의 각도와 속도 또는 보드 게임의 보드 상태 등이 있습니다.
- reward(*float*) : 입력 받은 action에 의해 달성된 reward를 반환합니다. 각 environmnet에 따라 reward의 상대적인 크기는 다르지만 합계 reward를 극대화하는 것이 목표입니다.
- done(*boolean*) : environment 를 reset 해야 할 때를 알려주는 결과값입니다. 목표에 도달하여 episode가 끝나게 되면 True를 반환합니다.
- info(*dict*) : 기본적으로 debugging에 필요한 정보들을 반환합니다.

아래의 그림은 고전적인 *agent-environment loop* 입니다. 각 timestep 마다 각 agent는 action을 선택하고, envrionment는 그렇게 선택된 action을 받아 observation과 reward를 agent에게 다시 알려줍니다.

이러한 과정은 초기 관찰값(initial observation)을 반환하는 reset() 함수를 호출하는 것에서부터 시작됩니다. 이후에는 (episode가 끝났음을 알려주는) *done* 값을 기준으로 반복하도록 하는 것이 일반적입니다.

```
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

### Spaces

위의 코드를 보면, environment 로부터 임의의 action 을 샘플링하고 있습니다(`action = env.action_space.sample()`). 실제로 environment 는 *action_space*와 *observation_space* 로 구성됩니다. 이들은 Space 타입이며, 각각 유효한 action과 observation 포맷을 알려줍니다.

```
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
```

*Discrete space*는 음이 아닌 실수의 범위를 가집니다. 그렇기 때문에 위의 경우 action들은 0 또는 1이 됩니다. 반면 *Box space*는 n 차원의 box를 의미한다. 따라서 유효한 observation은 4개의 수로 이뤄진 배열이 됩니다. 우리는 아래의 코드를 통해 Box의 범위 또한 확인할 수 있습니다.

```
print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])
```

이러한 정보를 이용하는 것은 서로 다른 환경에서 동작하는 generic code를 작성하는 데에 도움이 됩니다. *Box*와 *Discrete*는 가장 대표적인 *Space*이다. sample()함수를 통해 *Space* 로 부터 샘플링을 할 수도 있고, 다른 함수들을 통해 *Space*에 속해있는지 여부도 확인할 수 있습니다.

```
from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8
```

*CartPole-v0* 에서 하나의 action은 왼쪽으로 가는 힘을, 다른 하나는 오른쪽으로 가는 힘을 나타냅니다. 만약 더 좋은 학습 알고리즘을 가지고 있다면, 직접 각각의 숫자가 무엇을 의미하는지 알기위해 노력하지 않아도 될 것입니다.

## Available Environments

Gym은 쉬운 것부터 어려운 것까지 여러 environment 들이 들어있습니다. [링크](http://gym.openai.com/envs/#classic_control)를 통해 사용 가능한 environment 들을 확인할 수 있습니다.

- Classic control and toy text : 작은 크기의 문제들로, 출발점으로 삼기에 좋은 environment 입니다.
- Algorithmic : 숫자를 더하기, 순서 뒤집기 등 계산(computations)을 수행하는 것과 관련있는 문제들입니다.
- Atari : 고전적인 atari game들이 들어있습니다.
- 2D and 3D robots : 로봇 시뮬레이션 작업을 수행하게 됩니다. 이 작업들은 빠르고 정확한 로봇 시뮬레이션을 위해 설계된 MuJoCo 물리 엔진을 이용하게 됩니다. MuJoCo는 사적 소프트웨어이지만 무료 라이센스를 제공하고 있습니다.

### register

gym의 주된 목적은 공통된 인터페이스를 가지고 서로 비교가 가능하도록 버전화된 environmnet의 집합체를 제공하는 것입니다. *register* 를 이용해 현재 설치된 환경에서 사용할 수 있는 environment 를 확인할 수 있습니다.

```
from gym import envs
print(envs.registry.all())
#> [EnvSpec(DoubleDunk-v0), EnvSpec(InvertedDoublePendulum-v0), EnvSpec(BeamRider-v0), EnvSpec(Phoenix-ram-v0), EnvSpec(Asterix-v0), EnvSpec(TimePilot-v0), EnvSpec(Alien-v0), EnvSpec(Robotank-ram-v0), EnvSpec(CartPole-v0), EnvSpec(Berzerk-v0), EnvSpec(Berzerk-ram-v0), EnvSpec(Gopher-ram-v0), ...
```

register는 *EnvSpec* 객체 리스트를 반환하게 됩니다. *EnvSpec*은 시도 횟수와 최대 step의 수 등과 같이 환경과 관련된 구체적인 작업 내용을 정의하고 있습니다. 예를 들어 *EnvSpec(Hopper-v1)*는 뛰어다니는 로봇을 2차원에 시뮬레이팅하는 것을 목표로 하는 environment를 정의합니다. *EnvSpec(Go9x9-v0)* 은 9X9 사이즈의 바둑 게임을 정의합니다.

이러한 environment의 ID는 Opaque string으로 처리되어 있습니다. 유효한 비교가 가능하도록 하기 위해서는 성능에 영향을 미칠 수 있는 environment의 변화가 있어서는 안됩니다. 따라서 기존의 환경을 바꾸고 싶다면 새로운 version으로 대체하게 됩니다. 처음 gym에 도입된 environment들은 모두 *v0* 접미사를 가지고 있습니다. 따라서 기존 environment에 대한 대체재들은 *v1*, *v2* 등의 접미사를 가지게 될 것입니다.
