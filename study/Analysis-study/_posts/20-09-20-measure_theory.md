---
layout: post
title: Measure Theory
category_num : 1
---

# Measure Theory

- E. Kowalski의 책 Measure and Integral을 참고하여 작성했습니다.
- update at 2020.09.20

## Measurable Space

어떤 집합 $$X$$와 그에 대한 멱집합(family of subsets of $$X$$) $$\mathcal M$$이 있다고 할 때, $$X$$에 대한 **$$\sigma$$-Algebra**는 다음 세 가지 조건을 만족하는 경우로 정의된다.

$$
\eqalign{
& \ 1. \emptyset \in \mathcal M, X \in M\\
& \ 2. \text{If } Y \in \mathcal M, \text{ then, the complement set } X - Y \text{ is also in } M\\
& \ 3. \text{If } (Y_n) \text{ is any countable family of subsets } Y_n \in \mathcal M, \text{ then } \cup_{n \geq 1} Y_n \in \mathcal M \text{ and } \cap_{n \geq 1} Y_n \in \mathcal M
}
$$

이때 어떤 집합 $$Y \in \mathcal M$$인 경우를 두고 $$Y$$는 $$M$$에 대해 측정 가능하다(is measurable for $$\mathcal M$$)이라고 한다. 그리고 $$(X, \mathcal M)$$을 **Measurable Space**라고 부른다.

이를 이용하여 **Measurable Function**도 정의할 수 있다. 두 Measurable Space $$(X, \mathcal M), (X', \mathcal M')$$가 있다고 할 때, $$Y \in \mathcal M'$$에 있어 $$f^{-1}(Y) = \{ x \in X \lvert f(x) \text{ in } Y \}$$가 $$\mathcal M$$에 속하면 $$f : X \rightarrow X'$$는 $$\mathcal M$$과 $$\mathcal M'$$에 대해 Measurable하다고 한다.

## Borel $$\sigma$$-Algebra

어떤 집합 $$X$$와 그에 대한 어떤 부분집합들의 집합 $$A$$가 있다고 할 때, $$A$$를 포함하는 가장 작은 $$\sigma$$-Algebra를 $$A$$에 의해 만들어진 $$\sigma$$-Algebra라고 하고(**generated $$\sigma$$-Algebra**) $$\sigma (A)$$로 표기한다. 참고로 $$X$$에 대해 가장 큰 $$\sigma$$-Algebra는 위에서 언급한 $$X$$의 멱집합 $$M$$이 된다.

**Borel $$\sigma$$-Algebra**는 위상공간(Topology Space) $$(X, \mathcal T)$$에서 $$\mathcal T$$에 의해 만들어지는 $$\sigma$$-Algebra라고 할 수 있으며, $$\mathcal B_X$$로 표기한다. Borel Set이라는 표현도 자주 나오는데, Borel Set이란 Borel $$\sigma$$-Algebra의 원소를 말한다([위키](<https://ko.wikipedia.org/wiki/%EB%B3%B4%EB%A0%90_%EC%A7%91%ED%95%A9>)). 참고로 $$\mathcal T$$는 위상공간의 정의상 $$X$$의 멱집합의 부분집합이므로 위에서 언급한 $$A$$를 대신할 수 있다.

## Measure on a $$\sigma$$-algoebra

측정 가능한 공간 $$(X, \mathcal M)$$에서 Measure $$\mu : \mathcal M \rightarrow [0, +\infty]$$는 다음 두 가지 조건을 만족한다. 여기서 $$Y_n$$은 $$Y_n \in \mathcal M$$ 중에서 서로 pairwise disjoint 관계에 있는 것들을 말한다. 즉 $$Y_n$$ 들은 서로 교집합이 없어야 한다.

$$
\eqalign{
&1. \ \mu(\emptyset) = 0 \\
&2. \ \mu(\cup_n Y_n) = \Sigma \mu(Y_n)
}
$$

**Measure Space**는 Measurable Space $$(X, \mathcal M)$$과 그에 대한 Measure $$\mu$$ 세 가지로 구성된 $$(X, \mathcal M, \mu)$$로 정의된다. 그리고 $$\mu$$에 대해 $$\mu(X) < + \infty$$인 경우 **Finite-Measure**라고 부르며, 대부분의 경우에 Measure들은 Finite한 특성을 가진다. 대표적인 Finite-Measure로는 Probability Measure $$\mu(X) = 1$$이 있다.

## Lebesgue Measure

실수 공간 상에서 가장 많이 사용되는 Measure로는 **Lebesgue Measure $$\lambda$$**가 있다. 실수의 Borel $$\sigma$$-algebra에 대한 Measure로서 $$\lambda$$는 어떤 $$a \leq b$$에 대해 다음과 같이 정의된다.

$$
\lambda([a, b]) = b - a
$$

$$[0, 1]$$ 구간에 한정하여 정의된 Lebesgue Measure는 Probability Measure로 사용될 수 있으며 가장 기본적인 Lebesgue Mesaure이기도 하다.

### Example 1

$$x \in \mathcal R$$에 대해 $$N = \{ x \}$$ 또는 $$Q = [x, x]$$에 대해 $$\lambda(N) = \lambda (Q) = 0$$이 성립한다. 즉 countable set에 대해서는 $$\lambda$$는 0이다(negligible).

## Borel Measure

위상 공간 $$(X, \mathcal T)$$에 대해 $$X$$에 대한 **Borel Measure**는 $$X$$의 Borel $$\sigma$$-algebra의 Measure $$\mu$$이다. 그리고 **Regular Borel Measure**는 $$X$$의 Borel Measure $$\mu$$ 중에서 어떤 Borel Set $$Y \in X$$에 대해 다음 두 가지를 만족하는 경우를 말한다.

$$
\eqalign{
& \mu(Y) = \text{inf} \{ \mu(U) \lvert U \text{ is an open set containing } Y \} \\
& \mu(Y) = \text{sup} \{ \mu(K) \lvert K \text{ is a compact subset containing } Y \} \\
}
$$

## Integral of a step function

Measure Space $$(X, \mathcal M, \mu)$$ 상에서 정의되는 Step Function $$s: X \rightarrow C$$는 다음과 같이 정의된다.

$$
s = \Sigma_{i=1}^n \alpha_i \mathcal X_{Y_i}
$$

이때 $$\alpha_i \in C$$는 $$s$$에 의해 결정되는 서로 다른 값(distinct value)이고, $$Y$$는 $$X$$의 disjoint subset으로서 $$Y_i = \{ x \in X \lvert s(x) = \alpha_i \} \in \mathcal M$$와 같이 정의된다.

위와 같이 정의되는 Step Function $$s$$에 대해 $$s$$가 Non-Negative라면, 어떤 Measurable Set $$Y \in M$$에 대해 그것의 Measure는 다음과 같이 Integral을 사용하여 표현할 수 있다.

$$
\int_Y s(x) d \mu (x) = \Sigma_{i=1}^n \alpha_i \mu (Y_i \cap Y) \in [0, +\infty]
$$

우변의 식을 간단하게 풀이해보면, $$s$$의 값이 $$\alpha_i$$인 경우에 속하는 $$Y$$를 $$Y \cap Y_i$$로 정의하고 그 크기 $$\mu(Y \cap Y_i)$$만큼 $$\alpha_i$$에 대해 가중 평균한 것으로 Integral로 이해할 수 있다. 위의 식은 아래와 같이 다양한 방식으로 보다 간단하게 나타내기도 한다.

$$
\int_Y s d \mu, \qquad \int_Y s(x) d \mu, \qquad \int_Y s \mu,
$$

### Proposition with Integral of a step function

$$f \geq 0$$이 Step Function이라고 할 때 다음과 같은 것들이 성립한다.

- $$\int f d \mu = 0$$이라 함은 거의 대부분의 영역에서 $$f(x) = 0$$을 의미하고, $$\int f d \mu < +\infty$$이라 함은 거의 대부분의 영역에서 $$f(x) < +\infty$$을 의미한다.
- 만약 $$\mu(Y) = 0$$이라면 $$\int_Y f d \mu = 0 \text{ even if } f = + \infty \text{ on } Y$$ 가 성립한다($$0 \cdot \infty = 0$$).
- 만약 $$0 \leq f \leq g$$라면 $$\int_X fd\mu \leq \int_X gd\mu$$가 성립한다.
- 만약 $$Y \subset Z$$라면 $$\int_Y fd\mu \leq \int_Z fd\mu$$가 성립한다.

Non Negative Measurable Function인 경우에는 다음 또한 성립한다. 모든 Measurable Function은 $$X$$에 대한 것이다.

- Non Negative Measurable Function $$f,g$$에 대해 $$\int_X (f + g) d\mu = \int_X f d \mu + \int_X gd\mu$$가 성립한다.
- Non Negative Measurable Function Sequence $$\{ f_n \}$$에 대해 $$g(x) = \Sigma_{n \geq 1} f_n(x) \in [0, +\infty]$$로 $$g$$를 정의한다면, $$\int_X g d \mu = \Sigma_{n \geq 1} \int_X f_n d \mu$$가 성립한다.
