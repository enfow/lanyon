---
layout: post
title: Bellman Operator
---

# Bellman Operator

- update date : 20.01.15

## Bellman Equation to Bellman Operator

### Expected Bellman Equation

Expected Bellman Equation은 다음과 같다.

$$
v_\pi(s) = \Sigma_{a \in A} \pi (a|s) (R_s^a + \gamma \Sigma_{s' \in S}) P_{ss'}^a v_\pi(s')
$$

이때 어떤 policy $$\pi$$를 가정하면 다음과 같이 표현할 수 있다.

$$
P_{ss'}^\pi = \Sigma_{a \in A} \pi(a|s) P_{ss'}^a
$$

$$
R_s^\pi = \Sigma_{a \in A} \pi (a|s) R_s^a
$$

즉 policy $$\pi$$를 따를 때 state $$s$$에서 $$s'$$로 변화할 확률은 $$P_{ss'}^\pi$$이고, state $$s$$에서 받을 것으로 예상되는 reward는 $$R_s^\pi$$라는 것이다.

### Bellman Operator

Bellman Operator는 기본적으로 위와 같은 Bellman equation과 내용적으로는 동일하지만 수학적으로 보다 편리하게 사용하기 위해 Operator의 형태로 표현한 것이라고 할 수 있다. 

우선 어떤 state space $$S$$의 state 개수가 $$n$$개라고 하자. 그럼 위의 식을 다음과 같이 벡터 간의 연산으로 표현할 수 있다.

$$

\left\lbrack
\matrix{
    v_\pi(s_1) \cr
    . \cr
    . \cr
    . \cr
    v_\pi(s_n) \cr
}
\right\rbrack

=

\left\lbrack
\matrix{
    R_1^\pi \cr
    . \cr
    . \cr
    . \cr
    R_n^\pi \cr
}
\right\rbrack

+

\gamma

\left\lbrack
\matrix{
    P_{11}^\pi . . . P_{1n}^\pi \cr
    . . . . . \cr
    . . . . . \cr
    . . . . . \cr
    P_{n1}^\pi . . . P_{nn}^\pi \cr
}
\right\rbrack

\left\lbrack
\matrix{
    v_\pi(s_1) \cr
    . \cr
    . \cr
    . \cr
    v_\pi(s_n) \cr
}
\right\rbrack
$$

이를 좌변과 우변이 모두 n * 1 백터인 백터 연산으로 표현 가능한데,

$$
v_\pi = R^\pi + \gamma P^\pi v_\pi
$$

이를 bellman operator라고 하고 기호 $$\tau$$를 이용해 표현한다. 구체적으로 expected bellman operator와 bellman optimality operator는 다음과 같이 정의된다.

$$
\tau^\pi (v) = R^\pi + \gamma P^\pi v_\pi\\
\tau^*(v) = \max_{a \in A} (R^a + \gamma P^a v)
$$

Bellman operator는 수학적으로 $$\rm I\!R^n \rightarrow \rm I\!R^n$$로, 다시 말해 $$\rm I\!R^n$$ 공간의 어떤 한 점에서 다른 한 점으로 매핑한다는 의미를 가진다. 이와 같은 특성을 이용하면 operator theory의 unique fixed point 개념을 적용하여 모델의 수렴성 여부를 판단할 수 있다.

## Usage of Bellman Operator

### Contraction of Bellman operator

어떤 Bellman operator가 수축(contraction)한다는 것은 다음이 성립한다는 것을 의미한다.

$$

\text{for any policy} \ \pi \ \text{any initial vector} \ v, \\

\lim_{k \rightarrow \infty}(\tau^\pi)^k = v_\pi, \ \lim_{k \rightarrow \infty}(\tau^*)^k = v_* \\

\text{where} \ v_\pi \ \text{is the value of policy} \ \pi \ \text{and} \ v_* \ \text{is the value of an optimal policy} \ \pi_*

$$

즉, 수축한다는 것은 어떤 v vector에서 어떤 policy를 가지고 시작하더라도 무한히 반복하면 해당 policy의 $$v_\pi$$로 수렴하게 된다는 것을 의미한다.

이 내용은 contraction mapping theory를 통해 증명이 되며, 구체적으로는 wiki의 [Banach fixed point theorem](<https://en.wikipedia.org/wiki/Banach_fixed-point_theorem>)을 참조하면 좋을 것 같다.

## Reference

- [StackExchange의 Bellman Operator 글](<https://ai.stackexchange.com/questions/11057/what-is-the-bellman-operator-in-reinforcement-learning>)
- [Wiki의 contradiction mapping](<https://en.wikipedia.org/wiki/Contraction_mapping>)
