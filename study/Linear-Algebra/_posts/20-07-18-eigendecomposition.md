---
layout: post
title: 7. EigenDecomposition
category_num : 7
---

# EigenDecomposition

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book과 주재걸 교수님의 강의 인공지능을 위한 선형대수를 듣고 작성했습니다.
- update at : 2020.07.18

## EigenDecomposition

**고유값 분해(EigenDecomposition)**는 주어진 행렬의 특성을 찾아내기 위해 사용하는 방법 중 하나이다. 우리 말로 **고유벡터**, **고유값**으로 번역되는 **EigenVector**와 **EigenValue**는 고유값 분해의 결과이자 고유값 분해의 과정에서 찾아야하는 벡터와 스칼라 값이라고 할 수 있다.

고유값 분해는 정방행렬에 대해서만 가능하며 다음과 같은 공식을 만족하는 경우를 말한다.

$$
\boldsymbol{A}\boldsymbol{x} = \lambda\boldsymbol{x}
$$

여기서 $$\boldsymbol{v}$$가 고유벡터가 되고 $$\lambda$$가 고유값이 된다. 위의 식에서 한 가지 유추해 볼 수 있는 것은 행렬 $$\boldsymbol{A}$$를 곱해주는 선형 변환의 결과로 $$\boldsymbol{x}$$의 크기는 $$\lambda$$만큼 바뀌었지만 그 방향은 변하지 않는다는 것이다. 즉 어떤 행렬의 고유벡터란 해당 행렬을 곱하더라도 방향이 변하지 않는 벡터를 말한다고 할 수 있다.

### EigenDecomposition and Linear Dependence

그렇다면 행렬 $$\boldsymbol{A}$$가 있을 때 행렬의 고유벡터와 고유값은 어떻게 알 수 있을까. 이를 위해 위의 식을 다음과 같은 방정식과 같이 표현할 수 있다.

$$
\eqalign{
& \boldsymbol{A}\boldsymbol{x} - \lambda\boldsymbol{x} = \boldsymbol{0} \\
\rightarrow & \boldsymbol{A}\boldsymbol{x} - \lambda\boldsymbol{I}\boldsymbol{x} = \boldsymbol{0}\\
\rightarrow & ( \boldsymbol{A} - \lambda\boldsymbol{I} ) \boldsymbol{x} = \boldsymbol{0}\\
}
$$

위의 식의 가장 쉬운 답은 $$\boldsymbol{x}$$가 $$\boldsymbol{0}$$인 경우(trivial solution)일 것이다. 하지만 고유 벡터가 0인 경우는 의미가 없기 때문에 일반적으로 고려하지 않는다. 그런데 $$\boldsymbol{x} \neq \boldsymbol{0}$$와 $$( \boldsymbol{A} - \lambda\boldsymbol{I} ) \boldsymbol{x} = \boldsymbol{0}$$를 동시에 만족하기 위해서는 $$( \boldsymbol{A} - \lambda\boldsymbol{I} )$$가 선형 의존이어야 한다. $$( \boldsymbol{A} - \lambda\boldsymbol{I} )$$이 선형 독립이라는 것은 각 column 벡터가 다른 column 벡터로는 줄일 수 없는 차원 축을 가지고 있다는 것을 의미하는데 이렇게 되면 $$\boldsymbol{x} = 0$$인 경우에만 가능하기 때문이다.

역행렬의 유무와 관련지어 생각해보면 판별식 $$det ( \boldsymbol{A} - \lambda\boldsymbol{I} ) \neq 0$$인 경우, 즉 역행렬이 있는 경우라면 가능한 $$\boldsymbol{x}$$가 단 하나 밖에 없다는 것을 의미하고 이는 곧 $$\boldsymbol{x} = \boldsymbol{0}$$인 경우가 된다. 반면 $$det ( \boldsymbol{A} - \lambda\boldsymbol{I} ) = 0$$ 이라면 해가 없거나 무수히 많게 된다. 그런데 여기서 $$\boldsymbol{x} = \boldsymbol{0}$$라는 해가 존재한다는 것을 알고 있으므로 이 경우 해는 무수히 많다고 할 수 있다. 결과적으로 역행렬이 존재하지 않는 경우에만 $$\boldsymbol{0}$$외에 다른 해가 존재하게 된다.

정방행렬에서는

- 선형의존과 역행렬의 있다는 동치이다.
- 선형독립과 역행렬이 없다는 동치이다.

가 성립하므로 $$\boldsymbol{A}$$가 선형 독립이라면 $$\lambda\boldsymbol{I}$$를 잘 결정하여 $$( \boldsymbol{A} - \lambda\boldsymbol{I} )$$는 선형 의존이 되도록 하여야 한다. 혹은 $$det ( \boldsymbol{A} - \lambda\boldsymbol{I} ) = 0$$이 성립하여야 한다.

### EigenDecomposition and Null Space

어떤 행렬 $$\boldsymbol{A}$$의 **Null Space** $$\text{Nul}\boldsymbol{A}$$는 $$\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$$을 만족하는 $$\boldsymbol{x}$$의 집합을 의미한다. 이를 위의 고유값 분해 식에 적용하면 $$\text{Nul}( \boldsymbol{A} - \lambda\boldsymbol{I} )$$는 곧 가능한 고유 벡터 $$\boldsymbol{x}$$의 집합이 된다.

$$
\begin{bmatrix}
a_{11} && a_{12} \\
a_{21} && a_{22}\\
a_{31} && a_{32}\\
\end{bmatrix}

\begin{bmatrix}
x_1\\
x_2\\
\end{bmatrix}
=
\begin{bmatrix}
0\\
0\\
0\\
\end{bmatrix}
$$

위의 식에서 행렬 $$\boldsymbol{A}$$의 Null space는 위 식을 만족하는 $$x_1, x_2$$의 조합이라고 할 수 있다. 그런데 위의 식은 row 별로 분해하여 다음과 같아 나타낼 수 있다.

$$
\begin{bmatrix}
a_{11} && a_{12}
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1\\
x_2\\
\end{bmatrix}
= 0 \\
\begin{bmatrix}
a_{21} && a_{22}
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1\\
x_2\\
\end{bmatrix}
= 0 \\
\begin{bmatrix}
a_{31} && a_{32}
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1\\
x_2\\
\end{bmatrix}
= 0 \\
$$

벡터 간의 내적은 두 벡터가 수직인 경우에만 0이 되므로, 위의 세 식을 모두 만족하기 위해서는 벡터 $$x$$가 $$\boldsymbol{A}$$의 모든 row 벡터들과 수직이어야 한다. 정리하자면 $$\text{Nul}\boldsymbol{A}$$는 $$\boldsymbol{A}$$의 모든 row 벡터들과 수직인 벡터들의 집합이라고 할 수 있다. 그리고 $$( \boldsymbol{A} - \lambda\boldsymbol{I} )$$의 row 벡터들과 모두 수직인 $$\boldsymbol{x}$$가 $$\boldsymbol{A}$$의 고유벡터가 된다.

Null Space와 관련하여 한 가지 짚고 넘어갈 것이 있다면 Null Space는 전체 벡터 공간 $$R^n$$의 subspace로서 Row Space와 차원의 합이 전체 벡터 공간의 차원 $$n$$과 항상 같다는 것이다.

$$\text{dim} \ n = \text{dim} \ \text{Row}A + \text{dim} \ \text{Nul}A $$

### Characteristic Equation and Eigen Space

**특성 방정식(Characteristic Eqation)**은 다음과 같이 정의되며 $$\boldsymbol{A}$$의 고유값 $$\lambda$$를 구하기 위해 사용된다.

$$
det (\boldsymbol{A} - \lambda \boldsymbol{I}) = 0
$$

이를 만족하는 고유값 $$\lambda_i$$에 따라 $$\text{Nul}(\boldsymbol{A} - \lambda_i \boldsymbol{I})$$가 달리 정의되므로 고유값에 따라 고유벡터가 달라진다고 할 수 있다. 그리고 전체 3차원 공간 속에서 Row Space가 1차원인 경우 Null Space는 2차원이 되며 이렇게 되면 평면 상의 모든 벡터들이 고유 벡터가 된다. 따라서 하나의 고유값에 대해 복수의 고유 벡터가 존재할 수 있다. 참고로 고유 벡터가 정의되는 공간을 고유값 $$\lambda_i$$의 **Eigen Space**라고 한다.

### Summary

고유값 분해는 사용하기 위해서는 아래와 같은 조건들이 붙으며 따라서 제한적인 상황에서만 사용이 가능하다. **특이값 분해(Singular Vector Decomposition)**은 이러한 제한이 대부분 사라지므로 보다 다양한 형태로 사용된다.

- $$\boldsymbol{A}$$는 정방행렬이어야 한다.
- $$det (\boldsymbol{A} - \lambda \boldsymbol{I}) = 0$$를 만족하는 $$\lambda$$를 찾을 수 있어야 한다.

고유값과 고유벡터를 구하는 과정은 다음 두 단계로 나눌 수 있다.

1. 특성 방정식 $$det (\boldsymbol{A} - \lambda \boldsymbol{I}) = 0$$에 따라 가능한 $$\lambda$$를 찾는다.
2. 각각의 $$\lambda$$에 따라 고유벡터가 존재하는 공간 EigenSpace를 찾는다.