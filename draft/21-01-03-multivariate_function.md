---
layout: post
title: Differentiation of Multivariate Function
category_num: 2
---

# Differentiation of Multivariate Function

- 김홍종 교수님의 **미적분학 1,2**를 참고하여 개인 공부를 목적으로 작성했습니다.
- Update at: 2021.01.03

## Multivariate

다변수 함수(Multivariate Function)란 말 그대로 함수의 입력 변수가 여러 개인 경우를 말한다.

$$
f: U \subset \mathcal R^n \rightarrow \mathcal R
$$

공역이 $$m$$ 차원의 공간, 즉 함수의 출력이 벡터로 주어질 수도 있다.

$$
f: U \subset \mathcal R^n \rightarrow \mathcal R^m
$$

첫 번째 예시와 같이 출력이 스칼라 값으로 주어지면 **실수 함수(Real-Valued Function)**라고 하고, 벡터 값으로 주어지면 **벡터 함수(Vector-Valued Function)**이라고 한다.

## Continuous Function

어떤 함수가 주어져 있을 때 미분 가능하려면 그 함수는 연속 함수여야 한다. 조금 더 정확하게 말해 모든 미분 가능한 함수는 연속 함수이다. 미적분학에서 말하는 함수의 연속성(Continuity)이란 다음과 같이 정의된다.

$$
\eqalign{
&\text{Let } f: U \subset \mathcal R^n \rightarrow \mathcal R^m \text{ and } p \in U\\
& \text{If and only if } \lim_{x \rightarrow p} f(x) = f(p), \\
&\text{then the function } f \text{ is continuous on } p
}
$$

한 마디로 함수 $$f$$의 정의역 $$U$$의 한 점 $$p$$가 있다고 할 때 $$f(p)$$와 $$\lim_{x \rightarrow p} f(x)$$가 같으면 $$p$$에서 $$f$$는 연속이라고 할 수 있다. 그리고 함수 $$f$$가 연속이라고 하기 위해서는 정의역 $$U$$의 모든 $$x$$에 대해 연속이어야 한다. 

### Properties of Continuous Function

연속 함수는 기본적으로 다음과 같은 특성을 가진다.

- 상수 함수는 연속 함수이다.
- 두 연속 함수의 곱과 합 또한 연속 함수이다.
- 두 연속 함수의 합성 또한 연속 함수이다.

### Extreme Value Theorem

**최대 최소 정리(Extreme Value Theorem)**란 유계(Bounded)인 닫힌 집합(Closed Set)에서 정의된 연속 함수는 최대 값과 최소 값을 가진다는 것에 대한 정리로, 해의 존재성을 보장한다는 점에서 연속 함수의 중요한 특성 중 하나라고 할 수 있다. 정리의 결과는 해가 존재한다는 것으로 명확하지만 조건으로 제시된 내용 중 **유계(Bounded)**와 **닫힌 집합(Closed Set)**이라는 특성에 대해서는 정리할 필요가 있다.

우선 닫힌 집합이 무엇인지 정확하게 알기 위해서는 이를 정의하기 위해 사용되는 Open Ball(or Interval, Disk)에 대한 개념부터 알아야 한다.

#### Open Set & Closed Set

열린 공(Open Ball)이란 아래 식과 같이 어떤 점 P를 중심으로 하고 반지름이 r인 원을 의미한다. 만약 아래 식에서 정의역이 1차원 공간이라면 Open Interval이 되고, 2차원 공간이라면 Open Disk라고 부를 수 있지만 표현 상의 차이이지 의미는 동일하다.

$$
\mathcal B^n (p, r) = \{ x \in \mathcal R^n \lvert \ \lvert x - p \rvert  < r \}
$$

이를 기준으로 부분 집합 $$U$$와의 관계에 따라 $$\mathcal R^n$$ 상에 존재하는 모든 점들을 다음 세 가지로 나누어 볼 수 있다.

- 내점: 해당 점을 중심으로 하는 임의의 열린 공이 모두 $$U$$에 포함되는 경우
- 외점: 해당 점을 중심으로 하는 임의의 열린 공이 모두 $$U$$에 포함되지 않는 경우
- 경계점: 해당 점을 중심으로 하는 임의의 열린 공의 일부는 $$U$$에 포함되지만 일부는 그렇지 않은 경우

부분 집합 $$U$$에 포함되는 모든 점이 내점이라면 $$U$$는 열린 집합(Open Set)이라고 한다. 즉 Open Set에는 $$U$$의 경계점은 하나도 포함되어 있지 않다. 반대로 내점과 함께 $$U$$의 모든 경계점이 $$U$$에 포함되는 경우를 **닫힌 집합(Closed Set)** 이라고 한다. 

#### Bounded Set

유계 집합(Bounded Set)은 지름이 유한한 집합을 말한다. 이때 말하는 지름이란 다음과 같이 집합의 임의의 두 점 사이의 거리의 상한을 의미한다.

$$
\text{diam} U = \sup \{ \lvert p - q \rvert \lvert p, q \in U \}
$$

## Directional Derivatives

다변수 함수 미분의 특징 중 하나는 변화의 방향이 하나만 존재하는 일변수 함수와 달리 다변수 함수는 변화의 방향이 다양하다는 것이다. 따라서 변화의 방향에 따라 동일한 지점에서도 다양한 미분 값이 계산될 수 있다. 이때 어떤 점에서 특정 방향으로의 변화율을 **방향 미분(Directional Derivatives)**라고 한다.
