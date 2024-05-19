---
layout: post
title: Convex Optimization Introduction
category_num: 1
keyword: "[Optimization]"
---

# Convex Optimization Introduction

- update at : 20.10.28

## Convex Optimization

최적화로 번역되는 **Optimization**이란 주어진 제약 조건 하에서 최적의 경우를 찾는 것이다. 용어의 정의에 따르면 Optimization은 크게 두 가지로 나누어 볼 수 있는데, 하나는 주어진 제약 조건과 그것을 벗어나지 않으면서 최적의 경우를 찾고자 하는 대상이 되는 목적함수를 정의하는 것이고, 다른 하나는 정의된 목적함수와 제약조건에 따라 최적의 해를 찾는 것이다. 이를 각각 **Problem Design**과 **Problem Solving** 이라고 할 수 있다. Optimization을 한다고 하면 많은 사람들이 Problem Solving을 먼저 떠올리는 경향이 있지만 현실적으로는 정확하게 Problem Design을 수행하는 것 또한 매우 중요하다.

### Convex Optimization Problem Design

Optimization Problem Design은 수학적으로 말해 아래와 같은 형태를 갖는 적절한 **Optimization Formulation**을 도출하는 것을 말한다.

$$
\eqalign{
    & \min_x \ f(x)\\
    & \ s.t. \eqalign{
        & g(x) \leq 0 \\
        & h(x) = 0
    }
}
$$

그리고 **Convex Optimization**이란 Optimization Formulation의 특별한 형태로서, 다음과 같은 조건을 만족하는 경우를 말한다.

- $$f(x)$$가 **Convex Function**이다.
- $$g(x)$$가 **Convex Function**이다.
- $$h(x)$$가 **Affine Function**이다.
- $$x \in C$$일때 $$C$$는 **Convex Set**이다.

Convex Optimization이 전체 Optimization 문제들의 서브셋이라면 그 중에서도 유독 Convex Optimization이 집중적으로 각광받는 이유는 무엇일까. 당연하게도 여러 Convex Optimization이 가지는 여러 특성들 때문에 보다 문제를 쉽게 해결할 수 있기 때문이다. 따라서 일반적인 Optimization 문제도 Convex한 성질을 부여하여 해결하기도 한다.

## Convexity

Convex Optimization이 무엇인지 정확히 알기 위해서는 Convex Set과 Convex Function이 무엇인지 알아야 한다.

### Convex Set

**Convex**를 우리 말로 하면 '볼록'이 된다. 세미나에서 가장 기억에 남는 점 중 하나는 `Convex = 볼록`이라는 생각을 버리라는 것이었는데, 볼록이라는 표현 대신 다음과 같은 수식을 외우는 것이 보다 정확하다고 한다. 정확하게 어떤 집합 $$C$$가 **Convex**하다는 것은 다음과 같이 정의된다.

$$
\begin{multline}
\shoveleft \text{[Definition]} \\
\shoveleft \text{ The set } C \text{ is convex when } \\
\end{multline}\\
 \theta x + ( 1 - \theta) y \in C \qquad \forall x, y \in C  \qquad \text{ where } 0 \leq \theta \leq 1
$$

아래 이미지를 통해 보면 보다 직관적으로 수식의 의미를 이해할 수 있다. 왼쪽 이미지의 경우 집합의 어떤 a, b를 선택하더라도 두 원소를 잇는 선분 또한 집합에 포함되지만 오른쪽 이미지에서는 두 원소 a', b'를 통해 확인할 수 있듯 이를 항상 만족하지는 못한다. 이와 같이 집합의 어떤 두 원소를 잇는 선분이 항상 그 집합에 포함되는 경우를 Convex Set이라고 하고, 그렇지 않은 경우를 Non-Conex Set 이라고 한다.

<img src="{{site.image_url}}/study/convex_introduction_convex_set.png" style="width:36em; display: block; margin: 0px auto;">

참고로 **Convex Hull**이란 어떤 Non-Convex Set에 대해 그것을 모두 포함하는 가장 작은 Convex Set을 말한다. 위의 오른쪽 Non-Convex Set에 대한 Convex Hull은 다음 파란 점선과 같이 구해진다.

<img src="{{site.image_url}}/study/convex_introduction_convex_hull.png" style="width:15em; display: block; margin: 0px auto;">

### Convex Function

Convex Function는 수학적으로 다음과 같이 정의할 수 있다.

$$
\begin{multline}
\shoveleft\text{[Definition]}\\
\shoveleft \ \text{The function } f \text{ is convex function when } f \text{ satisfy } \\
\shoveleft f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta)f(y) \text{ where } \ 0 \leq \theta \leq 1, \forall x, y \in Dom f \\
\end{multline}
$$

위의 수식을 Convex Set과도 연관지어 생각해 볼 수 있는데, 한 마디로 정의하면 Epigraph가 항상 Convex Set 인 function $$f$$를 **Convex Function** 이라고 한다. 이 또한 이미지를 통해 확인하면 직관적이다.

<img src="{{site.image_url}}/study/convex_introduction_convex_function.png" style="width:22em; display: block; margin: 0px auto;">

### $$y = ax$$ is Convex Function

전혀 볼록하지 않아 보이는 1차 함수 $$y = x$$ 또한 위의 정의를 만족하므로 Convex Function 이다. 이러한 점에서 보면 Convex Optimization에서 말하는 Convex가 우리가 상상하는 '볼록'하다는 것에는 차이가 있으며, Convex Optimization에서 Convex의 의미를 볼록이라는 표현에 집중하기 보다는 Convex Set을 정의하는 수식으로 받아들이는 것이 적절하다.
