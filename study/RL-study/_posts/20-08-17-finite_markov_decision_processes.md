---
layout: post
title: Finite Markov Decision Processes
category_num: 3
---

# Finite Markov Decision Processes

- Sutton의 2011년 책 Reinforcement Learning: An Introduction 2nd edition을 참고해 작성했습니다.  
- update at : 2020.08.17

## Agent-Environment Interaction

강화학습은 학습의 주체이자 의사결정자인 Agent와 이를 둘러싸고 있는 외부 환경 Environment 간의 상호작용을 바탕으로 학습이 이뤄진다.

<img src="{{site.image_url}}/study/agent_environment_interation.png" style="width:35em; display: block; margin: 0px auto;">

위 그림은 Agent와 Environment 간의 상호작용을 보여주고 있다. Agent는 $$t$$ 시점에 State $$s$$에 따라 적절한 Action $$a_t \in A$$를 결정하여 Environment에 전달한다. 그럼 Environment는 그에 맞춰 Reward $$r_{t+1} \in R$$와 Next State $$s_{t+1} \in S$$을 정하여 Agent에게 알려주게 된다. 강화학습은 이와 같은 상호작용을 반복하며 누적 Reward(Cumulative Reward)를 극대화하는 방법을 찾아내는 것을 목표로 한다.

이와 관련하여 강화학습의 주요 키워드로는 State, Action, Reward, Return, Policy 등이 있다.

### State $$s_t$$

State란 Environment가 Agent에게 현재 상태가 어떠한지 알려주는 정보라고 할 수 있다. 이때의 State는 반드시 해당 순간의 처리되지 않은 정보일 필요는 없고 복잡한 과정을 거쳐 특징을 추출하거나, 과거의 정보들을 Sequence로 제공해주는 것도 가능하다. Agent는 Environment에 의헤 제공된 State를 바탕으로 Action을 결정하게 된다.

### Action $$a_t$$

Agent가 현재 State를 바탕으로 결정한 행동을 의미한다. Action의 형태는 다양하며 Agent가 임의로 통제할 수 있는 값이라면 모두 Action이라고 할 수 있다. 이러한 점에서 Agent와 Environment를 구분하는 경계는 Agent에 의한 자유로운 통제의 가능성에 있다.

### Policy $$\pi(a \lvert s)$$

Agent는 Environment로부터 현재 State를 전달받아 그에 맞춰 적절한 Action을 결정한다고 했었다. 즉 Agent는 내부적으로 State를 Action으로 매핑하는 기능을 가지게 되는데, 이를 Policy라고 부른다. 쉽게 말해 Policy는 State를 Action으로 매핑하는 함수이며 $$\pi(a \lvert s)$$로 표기한다. 그리고 강화학습의 목표는 이 Policy를 적절히 변화시켜 받을 수 있는 Reward의 총합을 극대화시키는 것이다.

### Reward $$r_t$$

Reward는 Scalar 값으로, 매 time step 마다 주어지며, 이전 시점에서의 State, Action 조합이 얼마나 좋았는지 알려주는 지표라고 할 수 있다. 실제 문제에 강화학습을 적용할 때 학습에 큰 영향을 미치는 요소 중 하나가 Reward Shaping, 즉 Reward를 언제 어떻게 줄 것인가이다. 이와 관련하여 몇 가지 유의해야 할 특성으로는 다음과 같은 요소들이 있다. 

- Reward는 Agent에 의해 계산되는 것이 아닌 Environment에 의해 주어지는 것이어야 한다.
- Reward는 우리가 진정으로 달성하고자 하는 것을 알려줘야 한다. 어떻게 해야하는지 지식을 알려주는 방향으로 설정하면 안 된다.
- Sub Goal을 여러 개 설정하면 비교적 쉬운 Sub Goal만 반복적으로 달성하고 Real Goal을 찾으려는 노력을 하지 않게 된다. 즉 Chess를 예로 들면 승리한 경우에만 Reward를 받아야 한다.

### Return $$G_t$$

특정 시점 $$t$$ 이후에 받을 것으로 기대되는 Reward의 총합을 Return $$G_t$$라고 한다. 강화학습은 경험을 통해 Agent가 Policy를 변화시켜 누적적인 Reward를 극대화하는 것이라고 했는데, 이때 누적 Reward는 현재 시점 이후에 받을 것으로 기대되는 Expected Return을 의미한다고 할 수 있다.

Return을 계산하는 방법은 크게 두 가지가 있는데 첫 번째 방법은 아래와 같이 단순히 모든 time step에서의 Reward를 더하는 **Simplest Sum Return**이다.

$$
G_t = r_{t+1} + r_{t+2} + ... + r_{T}
$$

여기서 $$T$$는 Terminal State로, 말 그대로 하나의 Episode가 끝나는 State를 의미하며 이에 도달하게 되면 Environment는 Starting State 또는 Starting State Distribution에 따라 임의로 결정한 State로 돌아가게 된다. 

Terminal State를 가지는 문제를 **Episodic Task**라 하고 그렇지 않은 문제를 **Continuing Task**라고 한다. 그런데 Countinuing Task의 경우 위에서 제시한 Simplest Sum 방법으로는 Return이 무한이 되어버린다는 문제가 있다. 따라서 아래와 같이 Discounted Factor $$\gamma$$를 도입해 시점에 따라 가중치를 달리하여 더하는 방법이 있다. 이러한 Return을 **Discounted Return**이라고 한다.

$$
\eqalign{
G_t &= r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+2} + ... = \Sigma_{k=0}^\infty \gamma^k r_{t+k+1}
}
$$

Discounted Factor $$\gamma$$는 1보다 작아야 $$G$$의 크기가 무한히 커지지 않으며, 크기가 커지면 커질수록 미래 Reward가 미치는 영향이 커진다. 그리고 $$\gamma = 0$$이면 현재의 Reward만 고려하게 된다. 실제 구현에서는 0.99를 많이 사용하는 편이다.

## Markov Property

강화학습에서 가장 중요한 가정 중 하나는 **Markov Property**라고 할 수 있다. Markov Property에 대해 [위키](<https://en.wikipedia.org/wiki/Markov_property>)에서는 다음과 같이 정의하고 있다.

- In probability theory and statistics, the term Markov property refers to the **memoryless property** of a stochastic process. ... A stochastic process has the Markov property if the conditional probability distribution of future states of the process (conditional on both past and present values) **depends only upon the present state**.

정리하자면 Markov Property란 확률 프로세스 중 현재 State만을 조건으로 미래 State 프로세스의 확률분포가 결정되는 특성을 말한다. 핵심적인 키워드는 **Memoryless**로, 현재 State 이전의 State들에 대해서는 알 필요가 없다는 점에서 Memoryless 라는 표현이 쓰이는 것으로 이해할 수 있다.

포탄의 예시를 생각하면 보다 이해하기 쉬운데, 어떤 궤적을 그리며 날아가는 포탄이 있다고 하자. 이때 포탄이 어느 방향으로 어떤 속도로 날아갈지 알아내기 위해서는 현재 위치와 현재 속도만으로도 충분하다. 과거에 어떤 궤적으로 날아왔는지에 대한 정보는 필요하지 않다. 이러한 특성을 가지는 문제를 Markov Property의 속성을 가진다고 한다.

Markov Property를 만족하면 아래 두 식이 완전히 동일해진다.

$$ 
\eqalign{
&1. \ Pr \{ r_{t+1}=r, s_{t+1}=s' \lvert s_0, a_0, r_1, ... r_t, s_t, a_t  \} \\
&2. \ Pr \{ r_{t+1}=r, s_{t+1}=s' \lvert s_t, a_t  \}
}
$$

Markov Property를 만족하면 현재 State와 Action 만으로도 충분히 그에 대한 Reward와 Next State를 알 수 있다. 이를 반대로 생각해보면 Markov State만으로도 최선의 Action을 선택하는 것이 가능하다는 것이 된다. 이와 같은 Markov Property는 현재 State만으로도 Action을 결정하고, Value를 알아낼 수 있도록 해준다는 점에서 강화학습에서 중요하다.

### Markov Decision Process(MDP)

Markov Property를 만족하는 강화학습 문제를 Markov Decision Process(MDP)라고 한다. MDP는 $$<S, A, P, R, \gamma>$$로 정의되는데 각각의 의미는 다음과 같다.

- $$S$$ : State Space
- $$A$$ : Action Space
- $$P$$ : State Transition Probability
- $$R$$ : Reward Space
- $$\gamma$$ : Discounted Factor

여기서 State, Action Space가 유한한 경우를 Finite MDP라고 하는데, Finite MDP는 전체 강화학습 문제의 90%를 차지한다. Finite MDP에서 State $$s$$, Action $$a$$와 Next State $$s'$$, Reward $$r$$의 관계는 다음과 같이 One-step dynamics로 정의할 수 있다.

$$
p(s', r \lvert s, a) = Pr \{ s_{t+1} = s', r_{t+1} \lvert s_t = s, a_t = a \}
$$

위의 식을 이용하여 Environment에 대해 다음 두 가지를 정의할 수 있다.

$$
\eqalign{
\text{Reward Function : }&r(s, a) = E[r_{t+1} \lvert s_t = s, a_t = a] \\
\text{State-Transition Probability : }&p(s' \lvert s, a) = Pr \{ s_{t+1} = s' \lvert s_t = s, a_t = a \} 
}
$$

### Value Function

**Value Function**이란 특정 State(또는 State- Action Pair)가 Agent에 있어 얼마나 좋은지 알려주는 함수를 말한다. 이때 좋은지 기준이 되는 것은 Expected Return 이다. 대부분의 강화학습 알고리즘은 이 Value Function을 정확하게 추정하는 방향으로 학습이 이뤄진다.

어떤 State의 정확한 Value를 알기 위해서는 다음에 어떤 Action을 취할 것인지 알아야 한다. 이러한 점에서 Value Function은 특정한 Policy 하에서 정의된다. State Value Function $$v$$는 특정 State가 얼마나 좋은지를 알려주는 함수로, 다음과 같이 구해진다.

$$
v_\pi (s) = E_\pi [G_t \lvert s_t = s] = E_\pi [\Sigma_{k=0}^\infty \gamma^k r_{t+k+1} \lvert s_t = s]
$$

특정 State에서 어떤 Action을 하는 것이 얼마나 좋은지 알려주는 Action Valun Function $$q$$는 다음과 같다.

$$
q_\pi (s, a) = E_\pi [G_t \lvert s_t = s, a_t = a] = E_\pi [\Sigma_{k=0}^\infty \gamma^k r_{t+k+1} \lvert s_t = s, a_t = a]
$$

 Environment와 최대한 많이 상호작용하고 이를 통해 Value function을 추정할 수 있다. 즉 여러 번 반복을 통해 value function의 seperate average를 구하고, 이를 value function의 값으로 사용하는 것이다. 이러한 방법을 **Monte Carlo**라고 한다.

### Bellman Equation 

Value Fucntion의 가장 큰 특징 중 하나는 아래와 같이 재귀적으로 Next State $$s'$$에 대한 value function으로 표현할 수 있다는 점이다. 이를 **Bellman Equation**이라고 한다.

$$
\eqalign{
v_\pi(s) 
&= E_\pi[G_t \lvert s_t = s] \\
&= E_\pi [\Sigma_{k=0}^\infty \gamma^k r_{t+k+1} \lvert s_t = s]\\
&= E_\pi [r_{t+1} + \gamma \Sigma_{k=0}^\infty \gamma^k r_{t+k+1} \lvert s_t = s]\\
&= E_\pi [r_{t+1} + \gamma G_{t+1} \lvert s_t = s]\\
&= E_\pi [r_{t+1} + \gamma v(s_{t+1}) \lvert s_t = s]\\
}
$$

Bellman Equation은 어느 State $$s_t$$의 Value를 Next State $$s_{t+1}$$의 기대 (Discounted) Value와 Reward $$r_{t+1}$$의 합으로 표현할 수 있게 해준다. 이에 따라 특정 시점의 State의 Value는 다음 State의 Value를 기준으로 업데이트 할 수 있다(Action Value에 대해서도 물론 가능하다). 이를 **Backup Operation**이라고 한다. Sutton은 책에서 Backup Operation을 강화학습 방법론의 심장이라고 표현하고 있다.

<img src="{{site.image_url}}/study/backup_diagram.png" style="width:35em; display: block; margin: 0px auto;">

### Optimal Value Function

어떤 Policy $$\pi$$가 다른 Policy $$\pi'$$보다 좋다고 하기 위해서는 다음과 같이 모든 State의 Value가 $$\pi'$$보다 커야 한다.

$$
\pi \geq \pi' \qquad \text{ if and only if } v_\pi(s) \geq v_{\pi'}(s) \text{ for all } s \in S
$$

다른 모든 Policy보다 좋은 Policy를 **Optimal Policy**라고 하고 $$\pi^*$$로 표기한다. 그리고 Optimal Policy를 따를 때 Value Function을 Optimal State-Value Function이라 하고 다음과 같이 정의된다.

$$
v^*(s) = \max_\pi v_\pi(s)
$$

Optimal Action-Value Function은 다음과 같다.

$$
q^*(s, a) = \max_\pi q_\pi(s, a)
$$

그리고 둘 사이에는 다음과 같은 관계가 성립한다.

$$
q^*(s, a) = E[r_{t+1} + \gamma v^*(s_{t+1}) \lvert s_t = s, a_t = a]
$$

마지막으로 어떤 state $$s$$에서 Optimal Action-Value Function $$q^*$$에 따라 그 값이 가장 큰 Action $$a$$를 결정하면 기대 Return을 극대화할 것으로 기대할 수 있다. 이러한 점에서 Optimal Policy $$\pi^*$$는 다음과 같이 정의할 수 있다.

$$
\pi^*= \eqalign{
& 1 \qquad \text{ if } \ a = \arg \max_a q^*(s, a) \\
& 0 \qquad \text{ otherwise}
}
$$

### Bellman Optimality Equation

Optimal을 가정하게 되면 위와 같이 State $$s$$에서 $$q(s, a)$$의 크기가 가장 큰 Action $$a$$를 선택하게 된다. 따라서 다음과 같은 식이 성립하는데 이를 **Bellman Optimality Equation**이라고 한다.

$$
\eqalign{
v_\pi(s)^* &= \max_{a \in A(s)} q_{\pi^*}(s, a) \\
&= \max_{a \in A(s)} \Sigma_{s', r} p(s', a \lvert s, a) [r + \gamma v^*(s')]
}
$$

$$q$$에 대해서도 다음과 같이 성립한다.

$$
\eqalign{
q^*(s, a) &= E[r_{t+1} + \gamma \max_{a'} q^*(s_{t+1}, a') \lvert s_t = s, a_t = a] \\
&= \Sigma_{s', r} p(s', r \lvert s, a) [r + \gamma \max_{a'} q^*(s',a')]

}
$$

Optimal에서의 Back-up Diagram은 다음과 같다.

<img src="{{site.image_url}}/study/optimality_backup_diagram.png" style="width:35em; display: block; margin: 0px auto;">

### Optimality and Approximation

강화학습 문제는 결국 Optimal Policy가 무엇인지 알아내는 과정으로 볼 수 있다. 하지만 Optimal Policy를 정확하게 구하기 위해서는 매우 많은 비용이 든다. 이와 관련해서는 구체적으로 다음 두 가지 문제를 생각해 볼 수 있다.

- Environment에 대해 완벽하고 정확하게 알고 있더라고 Optimal Policy를 구하는 것은 어렵다.
- 정확히 Optimal을 알아내기 위해서는 매우 큰 메모리가 필요하다.

이때 State의 개수가 적고 유한하다면 array, table 형태로 데이터를 저장하고 이를 바탕으로 Optimal Policy를 근사하는 것이 가능하다. 이를 **tabular case**라고 하며 이러한 방법으로 근사하는 것을 **tabular method**라고 한다.

Agent-Environment Interaction을 샘플링하고 이를 기준으로 근사하는 방법은 state의 방문 빈도에 따라 정확도가 달라질 수 있다는 문제에서 자유롭지 못하다. 방문할 확률이 낮은 state의 경우 그 값이 정확하지 않을 가능성이 높고, 이 경우 좋지 못한 행동을 할 가능성이 높아진다. 이러한 문제는

- 자주 방문하는 State에 대해서만 많이 학습하게 된다는 점 
- 방문 빈도가 낮은 경우 Sub Optimal을 선택하더라도 전체에 미치는 영향이 낮다는 점 

등으로 인해 발생한다. 이러한 문제 때문에 경우에 따라서는 프로들과만 바둑을 두며 높은 승률을 기록한 강화학습 알고리즘이 초보들과의 바둑에서는 승률이 그보다 낮을 수 있다.