---
layout: post
title: "Quick Review: Linear Algebra"
category_num : 0
---

# Quick Review: Linear Algebra

- 김홍종 교수님의 책 **미적분학 1,2** 및 주재걸 교수님의 강의 인공지능을 위한 선형대수를 듣고 작성한 이전 포스트를 바탕으로 작성했습니다.
- update at : 2021.01.02

## Keywords

- 벡터(Vector), 단위 벡터, 표준 단위 벡터
- L2 Norm, 유클리드 거리
- 선형 결합, 선형 독립, Span, 기저(Basis)
- 내적(Inner Product)
- 정사영(Orthogonal Projection)

## Vector

- **벡터는 크기와 방향에 대한 표현**으로, 크기와 방향이 동일하면 시점/종점과 무관하게 같은 벡터라고 할 수 있다(동등하다).
- 동등한 벡터 중 원점을 시점으로, $$\mathcal R^n$$의 한 점을 종점으로 하는 벡터를 대표로 사용한다. 이러한 점에서 벡터는 $$\mathcal R^n$$의 원소들과 1:1로 대응된다.
- **벡터의 크기**를 측정하는 함수를 **Norm**이라고 한다. 그 중 가장 대표적인 것은 L2 Norm으로 Euclidean Distance라고도 한다([Inner Product and Norm](<https://enfow.github.io/study/linear-algebra/2020/07/13/inner_product_and_norm/>)).

$$
\| v \|_2 = \root \of {v_1^2 + v_2^2 + ... + v_n^2}
$$

- **단위 벡터**는 크기가 1인 벡터로, 방향의 단위라고 할 수 있다. 크기가 1 이므로 어떤 벡터와 나란한 방향의 단위 벡터는 L2 Norm을 나누어주어 구할 수 있다.

$$
{v\over {\| v \|_2}}
$$

- 벡터의 Element 중 하나만 1이고 나머지는 모두 0인 단위 벡터를 **표준 단위 벡터**라고 한다.

## Linear Combination

- **선형 결합**(일차 결합, Linear Combination)이란 벡터에 스칼라 곱을 하거나 벡터 간 덧셈으로 새로운 벡터를 만드는 연산을 말한다.
- 임의의 벡터 $$a$$는 표준 단위 벡터 간의 선형 결합으로 나타낼 수 있다.

$$
\eqalign{
&i = (1,0,0) \qquad j = (0,1,0) \qquad k = (0,0,1) \\
&a = a_1 i + a_2 j + a_3k
}
$$

- 벡터 $$\boldsymbol{v_1}, \boldsymbol{v_2}, ... \boldsymbol{v_k}$$가 있다고 할 때 이들 간의 선형 결합으로 벡터 $$\boldsymbol{v}$$를 만들 수 없는 경우를 두고 **선형 독립**(일차 독립, Linear Independence)이라고 한다. 즉, 선형 독립이라고 하기 위해서는 다음 공식의 해가 없어야 한다. 반대로 해가 있는 경우를 **선형 종속**이라고 한다.

$$
\boldsymbol{v} = t_1 \boldsymbol{v_1} + t_2 \boldsymbol{v_2} + ... t_k \boldsymbol{v_k}
$$

- **스팬(Span)**이란 전체 공간의 부분 공간(Subspace)로서, 주어진 벡터들 간의 선형 결합으로 나타낼 수 있는 벡터들의 집합으로 정의된다([Linear Combination](<https://enfow.github.io/study/linear-algebra/2020/07/11/Linear_Combination/>)).
- 벡터가 하나인 경우에는 Span은 1차원으로 주어지고, 벡터가 두 개인 경우에는 최대 2차원으로 주어진다.
- 주어진 벡터가 서로 선형 독립이면 Span의 차원은 벡터의 개수만큼 커진다.
- 주어진 벡터와 선형 종속인 벡터가 새로 추가된다 하더라도 Span의 차원은 커지지 않는다.
- $$\mathcal R^n$$ 공간의 모든 벡터들은 서로 선형 독립인 $$n$$개의 벡터들의 선형 결합으로 나타낼 수 있다. 이때 $$n$$개의 벡터들이 $$\mathcal R^n$$ 공간을 생성한다고 하여 **기저(Basis)**라고 한다(`증명-미적분학1, 215`).

## Inner Product

- **벡터의 내적**(Inner Product)은 벡터의 기본 연산 중 하나로, 다음과 같이 정의되는 스칼라 값이다.

$$
a \cdot b = a_1 b_1 + a_2 b_2 + ... + a_n b_n
$$

- 어떤 벡터의 L2 Norm의 제곱은 자기 자신과의 내적과 같다.

$$
\| v \|_2^2 = v \cdot v
$$

- 내적은 기하학적으로 다음과 같은 의미를 가진다.

$$
a \cdot b = \| a \|_2 \| b \|_2 \cos \theta
$$

- 이와 관련하여 (1) 벡터의 크기가 일정하다면 방향이 비슷할수록 내적의 크기는 커진다. (2) 두 벡터가 모두 영벡터가 아니고, 내적이 0이라면 두 벡터는 직교한다. 라는 것을 알 수 있다([Inner Product and Norm](<https://enfow.github.io/study/linear-algebra/2020/07/13/inner_product_and_norm/>)).

## Orthogonality

- 벡터 $$a$$에 대한 벡터 $$b$$의 **정사영**(Orthogonal Projection)이란 벡터 $$b$$에서 벡터 $$a$$에 내린 수선의 발이라고 할 수 있다.
- 벡터 $$a$$와 방향이 동일하면서 벡터 $$b$$와 가장 가까운 벡터로 찾을 수 있다. 쉽게는 다음 공식에 따라 구할 수 있다. 이때 $${a \cdot b \over a \cdot a}$$를 벡터 $$b$$가 가지는 $$a$$ 성분 이라고 한다(`증명-미적분학1, 188`).

$$
p_a(b) = {a \cdot b \over a \cdot a} a
$$

## Linear Mapping

- 어떤 벡터에 행렬을 곱하면 벡터는 새로운 공간의 다른 벡터와 매핑된다. 이러한 점에서 행렬을 선형 사상이라고 한다.
- 행렬은 선형 사상이고, 모든 선형 사상은 행렬로부터 얻어진다. 다시 말해 행렬과 선형 사상은 같은 것이다.
- 예를 들어 $$m \times n$$ 행렬 $$A$$와 $$\mathcal R^n$$ 공간 상의 벡터 $$v$$를 곱한 결과는 $$\mathcal R^m$$ 공간 상의 한 벡터 $$v$$이 된다.

$$
L_A: \mathcal R^n \rightarrow \mathcal R^m
$$

- 행렬의 열백터(Column Vector)들은 벡터들이 매핑되는 공간의 기저(Basis)가 된다.