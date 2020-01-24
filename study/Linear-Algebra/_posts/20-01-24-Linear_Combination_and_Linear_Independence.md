---
layout: post
title: 2. Linear Combination and Linear Independence
category_num : 2
---

# Linear Combination and Linear Independence

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다. 따라서 선형대수를 깊게 다루지는 않고 머신러닝에 있어 중요한 내용들로 이뤄져 있습니다.
- update at : 20.01.24

## solution of equation

아래와 같은 방정식을 선형 방정식이라고 하는데,

$$
Ax = b
$$

모든 $$b$$에 있어 각각에 대해 단 하나의 해만 있어야 행렬 $$A$$에 대해 역행렬 $$A^{-1}$$이 존재한다.

$$
x = A^{-1}b
$$

하지만 이와는 달리 어떤 $$b$$에 대해 해가 하나도 존재하지 않거나, 무수히 많은 경우도 있다. 해가 무수히 많은 경우에는 해가 되는 vector $$x, y$$에 대해 아래의 $$z$$ 또한 방정식의 해가 된다.

$$
z = \alpha x + (1 - \alpha) y, \qquad any \ \alpha \in \rm l\!R
$$

즉, 행렬 $$A$$와 벡터 $$b$$의 특성에 따라 **선형 결합 방정식의 해가 없거나, 하나 있거나, 무수히 많을 수 있다.**

## linear combination

사전적으로 선형결합(linear combination)은 "스칼라배와 벡터들의 덧셈을 통해 조합하여 새로운 벡터를 얻는 연산"([wiki](<https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%EA%B2%B0%ED%95%A9>))으로 정의된다.

$$
Ax = b
$$

위와 같은 선형 방정식을 실제로 계산하면 다음과 같다.

$$
Ax = \Sigma_i x_i A_{:,i}
$$

이는 곧 행렬 $$A$$의 column vector 간의 선형 결합으로 vector $$b$$를 나타낸다는 것을 의미한다.

#### * Column vector

행렬을 구성하는 vector를 column vector라고 한다.

$$
\boldsymbol{x} =
\begin{bmatrix}
1,&& 2,&&3 \\
1,&& 2,&& 3 \\
1,&& 2,&& 3, \\
\end{bmatrix}
$$

이 있다고 할 때 column vector는

$$
\begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix}

\

\begin{bmatrix}
2 \\
2 \\
2 \\
\end{bmatrix}

\

\begin{bmatrix}
3 \\
3 \\
3 \\
\end{bmatrix}
$$

이라고 할 수 있다.

그리고 column vector로 표현할 수 있는 공간을 column space라고 한다.

### 선형 결합의 기하학적 의미

이때 행렬 $$A$$의 column vector를 특정 공간 상 origin, 즉 모든 element가 0인 지점에서 해당 vector가 가리키는 point까지의 이동방향으로 생각해볼 수 있다. 그럼 위의 식은 특정 point $$b$$까지 가기 위해 행렬 $$A$$의 column vector $$A_{:,i}$$를 $$x_i$$배 만큼 더하여 가는 방법을 의미한다. 그리고 가는 방법의 개수가 방정식 해(solution)의 개수가 된다.

만약 행렬 $$A$$가 3x2 행렬이고, $$b$$가 3 dimension vector라고 한다면, $$x$$는 2 dimension vector가 된다. 따라서 이 문제는 $$A$$의 두 column vector를 사용하여 만들 수 있는 2차원 평면(또는 1차원 직선) 상에 $$b$$가 놓여 있는 경우에만 해를 가진다고 할 수 있다.

따라서 $$b$$가 n차원의 vector인 경우 모든 point $$b$$가 해를 가지기 위해서는 적어도 행렬 $$A$$의 column 수가 n개 이상이어야 한다. 하지만 이것만으로도 충분하지 않은데, 왜냐하면 각 column vector가 동일한 방향을 나타낼 수도 있기 때문이다. 이와 관련된 개념 중 하나가 Linear Independence이다.

## Linear Independence

선형 독립은 어떤 vector들의 집합 $$V = {v_1, v_2, ..., v_n}$$이 있을 때, 모든 vector $$v$$에 대해 다른 vector들 간의 선형결합으로 만들 수 없는 경우를 말한다. 선형 독립 관계를 가진다고 할 때 각 vector들은 기저가 된다.

#### * 기저(basis vector)

기저란 "벡터 공간의 임의의 벡터에게 선형결합으로서 유일한 표현을 부여하는 벡터들"([wiki](<https://ko.wikipedia.org/wiki/%EA%B8%B0%EC%A0%80_(%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98%ED%95%99)>))이다. 다시 말해 어떤 공간에서 다른 vector로는 표현이 불가능하며, 하나의 축으로 기능하는 vector를 말한다.

기저의 개념을 생각할 때 $$b$$가 n차원의 vector인 경우 모든 point $$b$$가 해를 가지기 위해서는 적어도 행렬 $$A$$의 column 중 **기저의 수**가 n개 이상이어야 한다.

### 선형 독립과 선형 결합 해의 개수

$$
Ax = b
$$

행렬 $$A \in \rm l\!R^{nxm}$$와 벡터 $$b \in \rm l\!R^m$$에 있어 해 $$x \in \rm l\!R$$의 개수는 다음과 같이 정해진다.

##### 1. $$A$$의 column 개수와 $$b$$의 차원 수가 동일하고, 모든 column vector가 기저인 경우

이 경우 해의 개수는 1개가 된다. 즉 모든 column vector를 이용해야만 해를 구할 수 있다. 

$$\rightarrow$$ 행렬 $$A$$의 역행렬이 존재한다.

##### 2. $$A$$의 column 개수와 $$b$$의 차원 수가 동일하지만, 모든 column vector가 기저는 아닌 경우

이 경우 해가 있을 수도 있고, 없을 수도 있다. 즉 기저 벡터로 만들어지는 평면(hyperplane) 상에 $$b$$가 위치하는 경우에는 해가 존재하지만, 그렇지 않으면 해가 없다.

##### 3. $$A$$의 기저가 아닌 vector가 존재하며, 다른 기저로 $$b$$를 표현할 수 있는 경우

이 경우 해가 무한히 많다. 즉 독립이 아닌 vector가 존재하고, 이 vector가 곱해지는 $$x$$ element값의 개수만큼(무한) 가능한 해가 존재한다.

## Inverse matrix and linear combination

모든 $$b$$에 있어 각각에 대해 단 하나의 해만 있어야 행렬 $$A$$에 대해 역행렬 $$A^{-1}$$이 존재한다고 할 수 있다고 했었다. 이를 위해서는 $$A$$의 모든 column vector들은 서로 선형 독립, 즉 기저여야 하고, column의 개수는 벡터 $$b$$의 차원의 개수와 같아야만 한다.

이러한 점을 고려할 때 행렬 $$A$$는 row의 수와 column의 수가 같은 정방행렬(square matrix)이고, 모든 column vector가 선형독립이면 역행렬이 존재한다. 즉, 이 두 조건을 만족해 역행렬이 존재하는 행렬을 가역행렬이라고 한다.

### 가역행렬(Invertible Matrix)

역행렬이 존재하는 행렬을 가역행렬이라고 한다. 가역행렬은 **정칙행렬(regular matrix)**, **비특이 행렬(non-singular matrix)**라고도 불린다. 반대로 역행렬이 존재하지 않는 행렬을 **특이 행렬(singular matrix)**라고 부른다.

어떤 행렬이 역행렬을 가지기 위해서는 위에서 $$Ax = b$$는 유일한 해 $$x = A^{-1}b$$를 가져야 한다고 했었다. 그리고 이를 만족하기 위해서는 행렬이 정방행렬인 동시에 모든 column vector가 선형 독립임을 만족해야 한다는 것을 보였다. 이를 보다 쉽게 판단하는 방법으로는 다음과 같은 공식이 있다.

$$
if \ \boldsymbol{x} =
\begin{bmatrix}
a, b \\
c, d
\end{bmatrix} and \ ad - bc \neq 0, \\
then, \ A \ have \ inverse \ matrix
$$

여기서 $$ad-bc$$는 행렬 $$A$$의 **행렬식(determinant)**이다.
