---
layout: post
title: 1. Element of Linear Algebra
category_num : 1
---

# Element of Linear Algebra

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다. 따라서 선형대수를 깊게 다루지는 않고 머신러닝에 있어 중요한 내용들로 이뤄져 있습니다.
- update at : 20.01.24

## 선형대수의 요소

#### 1. scalar

scalar는 단일 숫자를 말한다. 방향은 나타내지 않고, 크기만을 갖는다.

#### 2. vector

Deep Learning book에서는 vector를 "an array of numbers", 즉 숫자들의 배열이라고 정의한다. vector를 구성하는 각각의 숫자(element)들을 제각기 다른 축에서의 크기라고 생각하면 vector는 어떤 공간 상에서의 한 점(point)를 의미한다.

$$
\boldsymbol{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
... \\
x_n
\end{bmatrix}
$$

#### 3. matrix

matrix는 "a 2-D array of numbers"로 정의된다. 즉 여러 개의 vector를 붙여둔 것이라고 이해할 수 있다. 2차원이므로 matrix는 높이와 너비를 가지는데, 이때 높이가 m이고 너비가 n인 matrix $$A$$를 $$A \in \rm l\!R^{mxn}$$으로 표기한다.

$$
\boldsymbol{x} =
\begin{bmatrix}
x_{1,1} && x_{1,2} && ... && x_{1,n}\\
x_{2,1} && x_{2,2} && ... && x_{2,n}\\
... && ... && ... && ... \\
x_{m,1} && x_{m,2} && ... && x_{m,n}\\
\end{bmatrix}
$$

참고로 scalar는 소문자, vector는 두꺼운 소문자, matrix는 대문자로 표기한다.

#### 4. tensor

tensor는 2차원 이상으로 이뤄진 배열을 말한다. tensor의 일반적인 형태로 [regular grid](<https://en.wikipedia.org/wiki/Regular_grid>)를 제시하고 있다.

## Transpose

전치행렬(transpose matrix)이란 다음과 같이 행과 열을 뒤집은 행렬을 말한다. 정확하게는 주 대각선을 축으로 반사 대칭을 가하여 얻은 행렬([wiki](<https://ko.wikipedia.org/wiki/%EC%A0%84%EC%B9%98%ED%96%89%EB%A0%AC>))이다. 기존 행렬에 T를 씌워 표현한다.

$$
A =
\begin{bmatrix}
A_{1,1} && A_{1,2}\\
A_{2,1} && A_{2,2}\\
A_{3,1} && A_{3,2}\\
\end{bmatrix}
$$

$$
A^T =
\begin{bmatrix}
A_{1,1} && A_{2,1} && A_{3,1}\\
A_{1,2} && A_{2,2} && A_{3,2}\\
\end{bmatrix}
$$

## matrix 간의 곱셈

행렬과 벡터를 곱한다고 하면 아래 두 가지 방법으로 가능하다.

#### 1. dot product

dot product는 scalar product, 내적(inner product)이라고도 하며 가장 기본적인 곱셈 방법이다. 즉, 일반적으로 선형대수에서 matrix 또는 vector를 곱한다고 하면 dot product를 수행한다고 생각해도 된다.

앞에 위치한 행렬의 각 row와 뒤에 위치한 행렬의 각 column을 서로 곱하고 모두 더해 하나의 element를 구성하는 방식으로 수행된다. $$A, B, C$$ 세 개의 행렬이 있다고 할 때 $$C = AB$$를 수학적으로 표현하면 다음과 같다.

$$
C_{l,j} = \Sigma_k A_{i,k}B_{k,j}
$$

따라서 dot product가 이뤄지려면 앞에 위치한 행렬 $$A$$의 column의 크기와 뒤에 위치한 행렬 $$B$$의 row의 크기가 일치해야 한다. 만약 $$A \in \rm l\!R^{mxn}$$, $${B \in \rm l\!R^{nxp}}$$라면 $$C$$는 $$C \in \rm l\!R^{mxp}$$가 된다.

벡터 간 dot product도 가능한데, 이를 위해서는 어느 한 벡터를 전치해주어야 한다. 즉, 다음과 같이 가능하다.

$$x^T y$$

이 경우에도 서로 크기가 맞아야하므로, $$x,y$$는 동일한 크기의 벡터여야 한다.

#### 2. element-wise product

element-wise product는 이름과 같이 앞의 행렬과 뒤의 행렬를 곱할 때 서로 동일한 위치의 element를 곱하는 것을 의미한다. 일반적으로는 사용되지 않으므로 $$A \bigodot B$$와 같이 특별한 표기법을 사용한다.

동일한 위치의 element 간의 곱이므로 앞의 행렬과 뒤의 행렬의 크기가 서로 일치해야 한다.

#### Matrix간 곱셈과 transpose

참고로 행렬의 곱셈과 전치 간에는 다음이 성립한다.

$$
(AB)^T = B^T A^T
$$

## Matrix 간 곱셈과 그 성질

#### 1. Distributive

행렬 간의 곱셈은 분배법칙이 성립한다.

$$
A(B+C) = AB + AC
$$

#### 2. Associative

행렬 간의 곱셈은 결합법칙이 성립한다.

$$
A(BC) = (AB)C
$$

#### 3. Commutative

하지만 교환법칙은 성립하지 않는다. 정확하게 말하면 항상 성립하는 것은 아니다.

$$
AB \neq BA
$$

단, 두 vector 간의 곱은 교환법칙이 성립한다.

$$
x^Ty = y^Tx
$$

## matrix inversion

우리말로 하면 역행렬인데, 역행렬을 이해하기 위해서는 우선 identity matrix(단위 행렬)에 대해 먼저 알아야 한다.

#### 1. 단위행렬(Identity matrix)

단위행렬이란 행렬의 주 대각선 요소가 모두 1이고, 다른 요소들은 모두 0인 행렬을 말한다. $$I$$로 표기한다.

$$
\boldsymbol{I} =
\begin{bmatrix}
1 && 0 && 0\\
0 && 1 && 0\\
0 && 0 && 1\\
\end{bmatrix}
$$

단위행렬은 행렬간 곱셈 연산에 있어 항등원의 성질([wiki](<https://ko.wikipedia.org/wiki/%EB%8B%A8%EC%9C%84%ED%96%89%EB%A0%AC>))을 띈다.

$$
AI = IA = A
$$

#### 2. 역행렬(matrix inversion)

역행렬은 다음과 같은 성질을 갖는 행렬을 말한다.

$$
A^{-1} A = I
$$

즉 어떤 행렬 $$A$$가 주어져 있을 때 곱해서 단위 행렬이 되는 행렬을 역행렬이라고 한다.

역행렬이 존재한다면 여러가지 알고리즘을 통해 이를 구할 수 있으며, 역행렬을 이용하면 $$Ax = b$$와 같은 문제의 해를 쉽게 찾을 수 있다는 장점이 있다. 하지만 실제 컴퓨터를 통한 연산에는 한계가 있어 자주 사용되지는 않고 이론적으로만 많이 사용된다고 한다.

#### 3. 역행렬의 성질

행렬 $$A,B,C$$ 모두 역행렬이 존재한다고 할 때 다음과 같은 공식이 성립한다.

- $$(A^{-1})^T = (A^T)^{-1}$$, 역행렬의 전치행렬은 전치행렬의 역행렬과 같다.
- $$(ABC)^{-1} = C^{-1}B^{-1}A^{-1}$$, 행렬 간의 곱에 대한 역행렬은 각 행렬의 역행렬을 역순으로 곱한 것과 같다.

## special kinds of matrices and vectors

### 대각행렬(Diagonal matrix)

대각행렬이란 주대각선의 요소 외에 다른 요소들은 모두 0인 행렬을 말한다. 주대각행렬의 요소가 모두 1인 identity matrix $$I$$ 또한 대각행렬의 일종이라고 할 수 있다. $$diag(v)$$로 대각행렬을 표기하기도 하는데 이는 정방행렬 중 주대각행렬의 요소가 벡터 $$v$$인 경우를 의미한다.

대각행렬은 계산상 편의성이 많은데, 대각행렬과 벡터 간의 곱은 $$diag(v)x = v \bigodot x$$로 쉽게 계산된다. 또한 대각행렬의 요소가 모두 0이 아닌 경우에는 역행렬의 계산은 아래와 같다.

$$
diag(v)^{-1} = diag([1/v_1, 1/v_2, ... ,1/v_n]^T)
$$

### 대칭행렬(symmetric matrix)

대칭행렬이란 다음과 같이 전치를 하여도 동일한 행렬을 말한다.

$$
A = A^T
$$

### 단위 벡터(unit vector)

단위 벡터란 벡터의 크기(L^2 norm)가 1인 경우를 말한다.

$$
\| x \|_2 = 1
$$

단위 벡터를 구하기 위해서는 전체 벡터의 요소 합을 각 요소에 나누게 되는데, 이러한 점에서 정규화 벡터(normalized vector)라고도 한다.

### 직교(orthogonal)성

#### 직교의 의미

벡터 $$x$$와 $$y$$가 서로 **직교(orthogonal)**한다는 것은 벡터 $$x$$와 $$y$$ 간에

$$x^Ty = 0$$

이 성립한다는 것을 의미한다. 기하학적으로 두 벡터가 서로 직각이기 때문에 직교라는 표현을 사용한다. n차원 공간에 정의된 벡터는 최대 n개의 직교행렬을 가질 수 있다.

직교하면서도 두 벡터의 크기가 1인 경우를 **정규직교(orthonormal)**의 관계를 가진다고 말한다.

#### 직교행렬

**직교행렬(orthogonal matrix)**이란 정방행렬 중 행벡터와 열벡터가 각각 다른 행벡터, 다른 열벡터와 정규직교하는 행렬을 말한다.

이러한 직교행렬은

$$
A^TA = AA^T = I
$$

의 특성을 가지는데, 이는 곧

$$
A^{-1} = A^T
$$

를 의미한다.
