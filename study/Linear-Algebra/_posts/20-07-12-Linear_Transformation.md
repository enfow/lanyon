---
layout: post
title: 3. Linear Transformation
category_num : 3
---

# Linear Transformation

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book과 주재걸 교수님의 강의 인공지능을 위한 선형대수를 듣고 작성했습니다.
- update at : 2020.07.12

## Subspace & Basis

**Subspace**과 **Basis**는 다음과 같이 정의된다.

- A subspace H is defined as a subset of $$R^n$$ closed under linear combination
- A basis of a subspace H is a set of vectors that satisfies both of the following
    - Fully spnas the given subspace H
    - Linear Independent

subspace란 어떤 vector 집합에 포함되어 있는 모든 vector 간 선형 결합에 대해 닫혀있는 벡터 집합을 말한다. 여기서 선형 결합에 닫혀있다는 것은 집합 내의 어떤 vector 간에 선형 결합을 하더라도 집합 내에 포함되어 있는 vector가 구해진다는 것을 의미한다. 그리고 **기저(Basis)**는 이러한 subspace를 선형 결합을 통해 완전히 커버할 수 있는(span) 선형 독립 벡터들의 집합이 된다.

아래 두 예시의 경우 모두 기저의 조건을 만족하지 못한다. 구체적으로 왼쪽 예시는 두 vector가 subspace를 모두 커버하지 못하기 때문이고 오른쪽 예시에서는 세 개의 vector가 선형 의존의 관계를 가지기 때문이다.

<img src="{{site.image_url}}/study/basis2.png" style="width:43em; display: block; margin: 0px auto;">

기저는 subpace에 대해 유일하지 않고, 아래와 같이 기저의 조건만 만족한다면 여러 조합이 존재할 수 있다.

<img src="{{site.image_url}}/study/basis1.png" style="width:43em; display: block; margin: 0px auto;">

참고로 Subspace의 차원은 기저 벡터의 갯수로 정의된다.

### standard basis

n 차원의 subspace를 표현하기 위해 사용되는 표준 기저는 다음과 같이 어느 한 성분만 1이고 나머지는 모두 0인 기저 벡터들의 집합를 말한다. 3차원의 경우 다음과 같다.

$$
\begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix}, \ \begin{bmatrix}
0 \\ 1 \\ 0
\end{bmatrix}, \ \begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix}
$$

표준 기저를 모두 concat하여 표현하면 Identity Matrix와 동일해진다.

## Linear Transformation

앞에서 본 Linear Combination은 주어진 벡터 간에 벡터 연산을 통해 새로운 벡터를 만들어내는 것이었으며 이 경우 Subspace는 변화하지 않았다. **선형 변환(Linear Transformation)** 은 어떤 한 벡터를 다른 벡터로 변환해주는 것을 의미하는데 그 과정에서 Subspace가 변화할 수도 있다. 즉 2차원 평면 상의 벡터가 선형 변환을 통해 3차원 공간 상의 벡터로 변화할 수 있다는 것이다. 물론 **선형** 변환인 만큼 이 과정에서도 선형성의 조건을 만족해야 한다.

- 벡터 공간 $$V, W$$
- 선형 변환 $$T : V \rightarrow W$$
- 임의의 두 벡터 $$u, v \in V$$에 대하여 $$T(u+v) = T(u) + T(v)$$
- 임의의 스칼라 $$a \in K$$ 및 벡터 $$v \in V$$에 대하여 $$T(av) = a T(v)$$ [위키피디아](<https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%EB%B3%80%ED%99%98>)

### Matrix of Linear Transformation

선형 변환을 쉽게 하는 방법은 기저 벡터들의 선형 변환을 Matrix 형태로 만들어 변환하고자 하는 vector에 곱해주는 것이다. 예를 들어 2차원의 벡터 공간 $$V$$ 상의 vector $$v$$를 3차원의 벡터 공간 $$W$$의 벡터로 변환한다고 해 보자. 이때 $$V$$의 기저벡터가 다음과 같이 주어져 있고

$$
\begin{bmatrix}
1 \\ 0 \\
\end{bmatrix}, \ \begin{bmatrix}
0 \\ 1 \\
\end{bmatrix}
$$

그 선형 변환의 결과가 다음과 같다면

$$
T(\begin{bmatrix}
1 \\ 0 \\
\end{bmatrix}) = \begin{bmatrix}
3 \\ 1 \\ 2
\end{bmatrix}, \qquad
T(\begin{bmatrix}
0 \\ 1 \\
\end{bmatrix}) = \begin{bmatrix}
4 \\ -2 \\ 7
\end{bmatrix}
$$

vector $$v = \begin{bmatrix}
3 \\ 2 \\
\end{bmatrix}$$는 벡터 공간 $$W$$에 다음과 같이 $$\begin{bmatrix}
17 \\ -1 \\ 20
\end{bmatrix}$$로 매핑(Mapping)된다.

$$
\begin{bmatrix}
3 && 4 \\ 1 && -2 \\ 2 && 7
\end{bmatrix}
\begin{bmatrix}
3 \\ 2 \\
\end{bmatrix} =
\begin{bmatrix}
17 \\ -1 \\ 20
\end{bmatrix}
$$

참고로 여기서 사용되는 Matrix는 Identity Matrix $$I$$를 선형 변환한 결과로 쉽게 구할 수 있다.

## Linear Transformation and Nueral Network

뉴럴넷은 기본적으로 벡터와 행렬의 곱셈으로 표현된다. 물론 activation function 등에 의해 연산 과정에서 선형성은 보존할 수 없지만 weight를 곱하는 것은 결국 Matrix와 Vector를 곱해 새로운 Vector로 변환하는 것이므로 뉴럴넷에서 하나의 레이어를 통과하는 것은 선형 변환이라고 할 수 있다.

<img src="{{site.image_url}}/study/linear_transfomation_and_nueral_net.png" style="width:32em; display: block; margin: 0px auto;">
