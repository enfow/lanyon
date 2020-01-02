---
layout: post
title: 1. What is Convex Optimization
---

# What is Convex Optimization

- update : 2020.01.02
  
## Optimization problem

최적화 문제는 주어진 조건 하에서 선택 가능한 집합의 원소 중 어떤 기준에 가장 최적인, 또는 최적에 가장 가까운 원소를 찾는 문제를 말한다. 이를 수학적으로 표현하면 다음과 같이 나타낼 수 있다.

$$ \min_{x} \enspace \enspace f(x)
\qquad
\qquad
subject \enspace to \enspace
\eqalign {g_{i}(x) \leqq 0,\enspace i = 1...m \\
           h_{j}(x) = 0,\enspace j = 1...r \\
}$$

$$
f : \rm I\!R^n \rightarrow \rm I\!R  \\
g_i : \rm I\!R^n \rightarrow \rm I\!R  \\
h_j : \rm I\!R^n \rightarrow \rm I\!R  \\
$$

이때 최소화의 대상이 되는 f 를 cost function 또는 objectives라고 하며, subject to 뒤에 위치해 f를 최소화 하는 데 있어 제약 조건이 되는 g와 h를 각각 inequality constrint function, equality constraint function 이라고 한다. 참고로 subject to 는 s.t.로 줄여쓰기도 한다.

- $$f$$ : objectives
- $$g$$ : inequality constrint function
- $$h$$ : equality constrint function

함수 f의 정의역은 기본적으로 실수 전체인 $$x \in \rm I\!R$$ 이 된다. 이 중 제약 조건을 모두 만족하는 정의역을 feasible domain 이라고 한다. 즉 feasible domain이란 최적화 문제의 해답이 될 수 있는 후보들이라고 할 수 있다. 그리고 이 feasible domain 중에서 f를 최소화하는 원소를 optimal solution 이라고 하고, $$x^*$$ 로 표기한다.

## Convex optimization problem

최적화 문제에 있어 중요한 개념 중 하나가 convex 이다. 최적화 문제는 수학적으로 표현할 수 있는 대부분의 문제에 다양한 방법으로 적용할 수 있다. 하지만 항상 그 해를 찾을 수 있는 것은 아니며, 오히려 특수한 몇몇 경우에만 공식적으로 최적의 해를 찾을 수 있다고 보아야 한다. 그리고 그 특수한 경우의 대표적인 예가 convex optimization 이다.

컨벡스 최적화 문제는 위에서 설명한 기본적인 최적화 문제와 틀을 같이하되 다음의 조건을 만족해야 한다.

1. objective function $$f$$ 는 convex function 이다.
2. inequality constraint function $$g$$ 는 convex function 이다.
3. equality constraint function $$h$$ 는 affine function 이다.

즉 objective function 과 constraint function 이 convex와 affine이라는 특수한 꼴을 지닐 때 최적화 문제를 해결하는 방법을 컨벡스 최적화 문제라고 하는 것이다.

### convex function

convex function의 [정의](<https://en.wikipedia.org/wiki/Convex_function>)는 다음과 같다.

$$
\begin{multline}
\shoveleft f \ is \ called \ convex \ if: \\
\shoveleft for \ all \ x_1, x_2 \in X, \ for \ all \ t \in [0, 1]: \\
\ f(tx_1 + (1 - t)x_2) \le tf(x_1) + (1 - t)f(x_2)
\end{multline}
$$

위의 부등식의 의미를 좌표계에서 확인하면

<img src="{{site.url}}/image/study/convex_function.png" width = 700>

에서 $$(x_1, f(x_1))$$ 과 $$(x_2, f(x_2))$$ 를 잇는 선분이 $$[x_1, x_2]$$ 구간에서 그래프보다 크거나 같은 곳에 위치함을 의미한다([이미지 출처 wiki](<https://en.wikipedia.org/wiki/Convex_function>)).

### affine function

affine function의 [정의](<https://glossary.informs.org/ver2/mpgwiki/index.php?title=Affine_function>)는 다음과 같다.

$$
\begin{multline}
\shoveleft f \ is \ called \ convex \ if: \\
\shoveleft for \ all \ x_1, x_2 \in X, \ for \ all \ t \in [0, 1]: \\
\ f(tx_1 + (1 - t)x_2) = tf(x_1) + (1 - t)f(x_2)
\end{multline}
$$

convex function의 정의와 거의 같은데, 부등호가 아닌 등호라는 점에서 차이가 있다. 즉, affine function 은 위의 좌표계에서 선분과 그래프가 완전히 일치한다.

### Convex set and Affine set

참고로 Convex set과 Affine set의 정의는 다음과 같다.

#### Convex set

어떤 두 점 $$ x_1 $$, $$ x_2 $$ 이 집합 C의 원소일 때, 두 점을 잇는 선분 또한 C에 포함되면 집합 C는 convex set 이라고 한다.

point, line, hyperplane, ball 등이 대표적인 convex set 이다.

#### Affine set

실수의 부분집합 $$ A \subset \rm I\!R^n $$ 의 두 점 $$ x_1 $$, $$ x_2 $$ 이 있을 때, $$ x_1 $$, $$ x_2 $$ 를 지나는 직선 또한 $$A$$에 포함되면 집합 $$C$$를 affine set이라고 한다.

## Reference

- [모두를 위한 컨벡스 최적화](<https://wikidocs.net/book/1896>)
- [wiki - convex function](<https://en.wikipedia.org/wiki/Convex_function>) 
- [링크1](<https://glossary.informs.org/ver2/mpgwiki/index.php?title=Affine_function>)
