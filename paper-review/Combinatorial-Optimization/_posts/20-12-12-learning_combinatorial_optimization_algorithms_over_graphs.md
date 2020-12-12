---
layout: post
title: "Learning Combinatorial Optimization Algorithms over Graphs"
category_num: 20
keyword: '[NP-hard graph]'
---

# 논문 제목 : Learning Combinatorial Optimization Algorithms over Graphs

- Hanjun Dai, Elias B. Khalil, Yuyu Zhang, Bistra Dilkina, Le Song
- 2017
- [논문 링크](<https://arxiv.org/abs/1704.01665>)
- 2020.12.12 정리

## Summary

- Graph 구조를 가지고 있는 NP-Hard 문제를 해결하는 방법으로 (1) 제안하는 Graph Embedding Network 구조인 **Structue2Vec**로 개별 Node의 Feature Embedding을 추출하고, (2) 그에 따라 매 Step의 Action으로 하나의 Node를 선택하는 과정을 반복적으로 수행하는 강화학습 방법론을 제시한다.
- 알고리즘의 일반적인 성능 검증을 위해 세 가지 대표적인 NP-hard graph 문제인 Minimum Vertex Cover, Maximum Cut, Traveling Salesman Problem에 대한 실험을 진행했으며, 서로 다른 성격의 세 가지 실험 모두에서 일정 성능을 확보할 수 있었다.
- 전체 Graph를 고려한 Node Embedding을 추출하기 위해 반복적으로 Node Embedding을 업데이트한다. 그리고 Dealyed Reward로 인한 문제점을 줄이기 위해 Q-iteration Approach를 사용하여 Q function을 업데이트한다.

## NP-hard graph optimization problem

논문에서 제시하고 있는 NP-hard Graph 문제는 **Minimum Vertext Cover(MVC), Maximum Cut(MAXCUT), Traveling Salesman Problem(TSP)** 세 가지이다. 문제의 정의에 사용되는 Notation은 다음과 같다.

- Gpaph: $$G(V,E, \mathcal w)$$
- Set of Nodes: $$V$$
- Set of Edges: $$E$$
- Edge Weight fucntion: $$\mathcal w(u, v)$$
- Subset of Nodes: $$S$$

### Minimum Vertext Cover(MVC)

**MVC**는 Graph에서 최소한의 Node를 선택하여 Graph의 모든 Edge를 커버하는 방법을 찾는 것이다. 여기서 커버한다는 것은 모든 Edge가 선택된 Node들과 연결되어 있음을 의미한다.

$$
\eqalign{
    &\text{Given Graph } G, \text{ Minimize } \lvert S \rvert \\
    &\text{When } S \text{ covers all of the Edges } E  
}
$$

### Maximum Cut(MAXCUT)

**MAXCUT**은 어떤 Edges 집합의 Weight가 극대화되도록 하는 방법을 찾는 문제이다. 이 경우에도 선택 대상은 Edge가 아니라 Node가 되며, 선택한 Node의 집합인 $$S$$와 그렇지 못한 Node 집합인 $$V \setminus S$$를 서로 연결하는 Edge를 선택한 것으로 본다.

$$
\eqalign{
    &\text{Given Graph } G, \text{ find } S \text{ which is maximize } \Sigma_{(u, v) \in C} w(u, v)\\
    &\text{Where the cut-set } C \subset E \text{ is the set of edges with one end in } S \text{ the other end in }V \setminus S
}
$$

### Traveling Salesman Problem(TSP)

세 문제 중 가장 유명한 **TSP**는 Edge를 최대 한 번만 사용하며 모든 Node를 순회할 때(처음과 끝이 동일) 사용한 Edge의 weight의 총합을 최소로 만드는 문제이다.

## How to Solve: Greedy Algorithm

MVC와 MAXCUT은 전체 Node 집합 $$V$$ 중 최적의 부분 집합 $$S \subset V$$를 선택하는 문제이고, TSP는 모든 Node $$V$$를 줄세우는 최적의 방법을 찾는 문제라는 점에서 서로 다른 점을 가지고 있다. 따라서 세 가지 문제, 나아가 Graph 구조의 NP-hard 문제에 대한 일반적인 해결책을 제시하기 위해서는 문제들이 공통적으로 가지는 요소를 찾고, 그에 맞춰 추상화해야 한다. 

논문에서는 세 가지 문제 모두 Node를 선택해야 한다는 점에 초점을 맞추고 있으며, 논문에서 제안하는 알고리즘은 이미 선택된 Subset $$S$$와 전체 Graph에 대한 정보를 가지고 한 번에 하나씩 Subset $$S$$에 새로운 node를 추가하는 방식을 택하고 있다. 이러한 방법론은 Heuristic Algorithm 중 하나인 Greedy Algorithm에서 영향을 받았다고 한다.

- A generic greedy algorithm selects a node $$v$$ to add next such that $$v$$ maximizes an evaluation function, $$Q(h(s), v) \in \mathcal R$$, which depends on the combinatorial structure $$h(s)$$ of the current partial solution.

$$
S := (S, v^*), \text{ where } v^* = \arg \max_{v \in \bar S} Q(h(S), v)
$$

- The step is repeated untila termination criterion $$t(h(s))$$ is satisfied.

전체적으로는 이러한 방법론을 적용한다 하더라도, 어떤 Node를 선택하는 것이 최적인지는 문제의 특성에 따라 달라진다. 따라서 개별 문제의 특성에 따라 추가적으로 다음과 같이 정의하고 있다.

### Minimum Vertext Cover

- **Objective Function**: $$c(h(S), G) = - \lvert S \rvert$$
- **Terminal Criterion**: is all edges cover?

MVC의 목표는 최소한의 Node를 선택하는 것이므로, 목적 함수는 Node 개수가 늘어감에 따라 줄어들게 된다. Node는 모든 Edge가 커버될 때까지 계속해서 선택하게 된다.

### Maximum Cut

- **Objective Function**: $$c(h(S), G) = \Sigma_{(u,v) \in C} w(u, v)$$
- **Terminal Criterion**: No need. 

MAXCUT의 목표는 Edge 집합의 Weight $$w$$의 총합을 극대화하는 것이며, 이 때 Edge 집합은 아래와 같이 정의된다.

$$C = \{ (u,v) \lvert (u,v) \in E, u \in S, v \in \bar S \}$$

### Traveling Salesman Problem

- **Objective Function**: $$c(h(S), G) = - \Sigma_{i=1}^{\lvert S \rvert - 1} \mathcal w (S(i), S(i+1)) - \mathcal w (S(\lvert S \rvert), S(1))$$
- **Terminal Criterion**: $$S = V$$

TSP는 하나씩 Node를 선택해가며 최종적으로 모든 Node를 선택해야 한다. 따라서 종료 조건은 $$S = V$$가 된다. 그리고 TSP는 마지막에 선택된 Node는 항상 시작 Node로 돌아와야 한다. 따라서 이를 고려해 목적 함수가 설계되어 있다.

## Graph Embedding: Structure2Vec

좋은 **Graph Embedding**이란 Graph의 전체적인 특성과 함께 개별 Node 혹은 Edge의 특성을 잘 표현해야 한다고 할 수 있다. 다음 Node를 Action으로 결정해야 하는 강화학습의 관점에서 본다면 Graph Embedding은 State의 질을 결정하게 되고, 이러한 점에서 문제의 특성을 정확하게 반영하는 Graph Embedding을 뽑아내야 좋은 Policy를 얻는 것 또한 가능하다.

### Structure2Vec

Graph Embedding을 뽑는다고 하면 Node Embedding과 Edge Embedding 두 가지를 생각해 볼 수 있다. 논문에서 풀고자 하는 문제들은 모두 Node를 선택하는 것인 만큼 여기서는 Node Embedding에 집중하고 있는데, 논문에서 제시하는 네트워크 구조인 **Structure2Vec**을 사용하면 아래 수식에 따라 각 Node의 Embedding $$\mu_v$$를 업데이트하는 방식으로 전체 Node Embedding을 구하게 된다.

$$\mu_v^{(t+1)} \leftarrow F( x_v, \{ \mu_u^{(t)} \}_{u \in \mathcal N_{(v)}}, \{ \mathcal w(v,u) \}_{u \in \mathcal N_{(v)}}; \Theta )$$

위의 식은 아래와 같이 Model Parameter와 Non-linear function으로 표현할 수 있다.

$$
\mu_v^{(t+1)} \leftarrow \text{relu} ( \theta_1 x_v + \theta_2 \Sigma_{u \in \mathcal N (v)} \mu_u^{(t)}, \theta_3 \Sigma_{u \in \mathcal N (v)} \text{relu} (\theta_4 w(v, u)))
$$

이때 $$\mu \in \mathcal R^p$$라고 하면, 각 Model Parameter는 $$\theta_1 \in \mathcal R^p$$, $$\theta_2 \in \mathcal R^{p \times p}$$, $$\theta_1 \in \mathcal R^{p \times p}$$, $$\theta_4 \in \mathcal R^p$$의 크기를 가지고 있다.

### Recursive Update

Structure2Vec의 두 번째 항에서는 현재 Node $$v$$와 인접한 모든 Node의 Feature Embedding $$\Sigma_{u \in \mathcal N (v)} \mu_u^{(t)}$$을 더하고 있다. 세 번째 항에서는 연결되어 있는 모든 Edge의 Weight $$w$$를 더하고 있다. 이렇게 구한 두 번째와 세 번째 항은 모두 새로운 $$\mu_v^{(t+1)}$$을 구하는 데에 사용된다. 즉, 한 번 Structure2Vec 구조를 통과하여 업데이트된 Node Feature는 그 인접 Node와 Edge에 대한 정보를 모두 가지고 있다고 할 수 있다.

만약 위의 업데이트 과정을 두 번 반복한다면 주변의 Node와 Edge에 대한 정보 뿐만 아니라 2번 건너뛰어 도달할 수 있는 Node와 Edge에 대한 정보(2 Hop)까지 얻을 수 있을 것으로 기대할 수 있다. 즉 이와 같이 Recursive하게 $$T$$번 업데이트하면 T-Hop 떨어진 Node와 Edge에 대한 정보도 담게 되며, 논문에서도 이러한 방법을 사용해 Node Feature Embedding을 뽑고 있다.

## Policy: greedy policy with Q function

앞서 언급한대로 Policy는 step마다 하나의 Node를 Action으로 결정해야 한다. 이를 위해 추출한 Node Embedding $$\mu$$을 사용하여 **현재 상태(partial solution) $$S$$에서 개별 Node를 선택하는 것의 가치(q-value)**를 계산하는 **$$Q$$ function**을 정의하고, Q function의 출력 값을 기준으로 Node를 greedy하게 선택하는 방법을 사용한다.

$$
Q(h(S), v; \Theta) = \theta_5^\text{T} \text{relu} (\text{concat}[\theta_6 \Sigma_{u \in V} \mu_u^{(T)}, \theta_7\mu(v^{(T)})])
$$

## Training: Q-learning

이러한 Q function은 강화학습의 방법론(**Q-learning**)에 따라 업데이트하게 된다. Action과 State에 대해서는 위에서 여러 번 언급하였고, Action $$a_t$$에 따라 Next State $$s_{t+1}$$가 deterministic 하게 결정되기 때문에 Transition도 문제의 특성상 중요한 고려대상이 아니다. **Reward Function**은 다음과 같이 각 문제의 목적 함수 $$c$$를 통해 구할 수 있다고 한다.

$$
r(S, v) = c(h(S'), G) - c(h()s, G)
$$

여기서 $$c(h(S'), G)$$는 이번 step에서 선택된 새로운 Node $$v$$가 포함된 partial solution을 기준으로 결정한 cost를, $$c(h()s, G)$$는 그렇지 않은 partial solution을 기준으로 구한 cost를 의미한다. 이와 같이 정의하고 있으므로 매 step의 Reward 총합 $$R$$은 마지막의 cost $$c(h(\hat S), G)$$와 동일하다.

$$
R(\hat S) = \Sigma_{i=1}^{\lvert \hat S \rvert} r(S_i, v_i) = c(h(\hat S), G)
$$

Reward Function까지 정의하였으므로 아래와 같은 기본적인 Q-learning의 업데이트 식에 따라 Q Function을 업데이트할 수 있다.

$$
\eqalign{
&(y - \hat Q (h(S_t), v_t; \Theta))^2\\
& \text{where } y = \gamma \max_{v'} \hat Q (h(S_{t+1}), v'; \Theta) + r(S_t, v_t)
}
$$

한 가지 아쉬운 점이 있다면 위와 같이 1 step 이후만을 기준으로 업데이트하게 되면 delayed reward 문제에 취약하다는 것이다. 쉽게 말해 장기적으로 이익이 되는 선택을 하지 않고, 근시안적으로만 Node를 선택하게 되어 Optimal Solution에 가까워지기 어려울 수 있다는 것이다. 이러한 문제를 줄이기 위해 다른 강화학습 알고리즘에서 쓰이는 **n-step Q-learning** 기법을 여기서도 제시하고 있다. n-step Q-learning는 업데이트 식은

$$(y - \hat Q (h(S_t), v_t; \Theta))^2$$

로 1-step Q-learning과 동일하며, $$y$$의 계산 방식이 아래와 같이 $$t + n - 1$$ 시점까지 고려한다는 점에서 차이가 있다.

$$
y = \Sigma_{t=0}^{n-1} r(S_{t+i}, v_{t+i}) + \gamma \max_{v'} \hat Q (h(S_{t+n}), v' ; \Theta)
$$

이를 구현하기 위해 Replay Buffer에 저장되는 Transition 또한 아래와 같이 일반적인 1-step Q-learning과 차이가 있게 된다.

$$
\eqalign{
&(S_t, a_t, R_{t,t+n}, S_{t+n}) \\
&\text{where } R_{t,t+n} = \Sigma_{i=0}^{n-1} r(S_{t+i}, a_{t+i})
}
$$

## Algorithm

최종적인 알고리즘은 아래와 같다.

<img src="{{site.image_url}}/paper-review/learning_combinatorial_optimization_algorithm_over_graph_algorithm.png" style="width:48em; display: block; margin: 0em auto; margin-top: 1em; margin-bottom: 1em">
