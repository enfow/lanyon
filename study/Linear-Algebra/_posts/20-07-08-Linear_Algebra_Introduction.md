---
layout: post
title: 1. Linear Algebra Introduction
category_num : 1
---

# Linear Algebra Introduction

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다. 따라서 선형대수를 깊게 다루지는 않고 머신러닝에 있어 중요한 내용들로 이뤄져 있습니다.
- update at : 2020.07.08

## "Linear"

선형대수에서 선형(Linear)은 "직선의 형태"를 말한다. 즉 선형대수란 직선의 형태를 띄고 있는 것에 관한 대수학이라고 할 수 있다. 수학적으로 **선형성(linearity)**이란 어떤 함수 $$f$$가 임의의 원소 $$x, y, a$$에 대해 다음 두 조건을 만족하는 경우를 말한다.

$$
\eqalign{
&1. \ f(ax) = af(x) \\
&2. \ f(x+y) = f(x) + f(y)
}
$$

---

예를 들어 1차 함수 $$y = 6x$$가 선형성을 가지는지 확인해본다면,

$$
\eqalign{
1. \ 6(ax) = a(6x)
2. \ 6(x+y) = 6x + 6y
}
$$

와 같이 두 가지 조건을 모두 만족한다. 따라서 선형성을 가진다고 할 수 있다.

---

반면 2차 함수 $$y = 2x^2$$의 경우

$$
\eqalign{
&1. \ 2(ax)^2 \ne a 2(x)^2 \\
&2. \ 2(x+y)^2 \ne 2x^2 + 2y^2
}
$$

두 가지 조건을 모두 만족하지 않으므로 선형성을 가진다고 할 수 없다. 2차 함수의 모양을 생각해 보더라도 선형과는 거리가 있어 보인다.

---

그런데 직선의 형태를 가지는 1차 함수의 경우에도 $$y = 6x + 2$$ 와 같이 원점을 지나지 않는 경우에는 선형성을 가지지 않는다.

$$
\eqalign{
& 1. \ 6(ax) + 1 \ne a(6x + 2) \\
& 2. \ 6(x+y) + 2 \ne 6x + 2 + 6y + 2
}
$$

즉 다항함수의 경우 1차함수 중에서도 원점 $$(0,0)$$을 지나는 경우에만 선형성을 가진다고 할 수 있다.

## Scalar, Vector and Matrix

선형대수의 기본적인 구성요소들로는 scalar, vector, matrix가 있다.

#### 1) scalar

scalar는 단일 숫자를 말한다. 방향은 나타내지 않고 크기만을 갖는다.

#### 2) vector

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

#### 3) matrix

matrix는 "a 2-D array of numbers"로 정의된다. 즉 여러 개의 vector를 붙여둔 것이라고 이해할 수 있다. 2차원이므로 matrix는 높이와 너비를 가지는데, 이때 높이가 m이고 너비가 n인 matrix $$A$$를 $$A \in \rm l\!R^{m \times n}$$으로 표기한다.

$$
\boldsymbol{A} =
\begin{bmatrix}
a_{1,1} && a_{1,2} && ... && a_{1,n}\\
a_{2,1} && a_{2,2} && ... && a_{2,n}\\
... && ... && ... && ... \\
a_{m,1} && a_{m,2} && ... && a_{m,n}\\
\end{bmatrix}
$$

참고로 scalar는 소문자, vector는 두꺼운 소문자, matrix는 대문자로 표기한다.

## Geometrical meaning of vector

선형대수는 벡터 공간을 가정하고 그 속에서 이뤄지는 벡터간 연산에 관한 것이라고 할 수 있다. 벡터 공간이란 쉽게 말해 좌표계인데 2차원 공간에서 벡터 $$
\begin{bmatrix}
1\\2
\end{bmatrix},
\begin{bmatrix}
2\\0
\end{bmatrix}
$$ 은 다음과 같이 크기와 방향을 가진 화살표로 표현된다.

<img src="{{site.image_url}}/study/2d_vector.png" style="width:22em; display: block; margin: 0px auto;">

3차원 공간에서도 가능하다. $$\begin{bmatrix}
1\\1\\2
\end{bmatrix},
\begin{bmatrix}
2\\2\\0
\end{bmatrix}
$$의 경우 아래 그림과 같다. 4차원, 그 이상도 가능하지만 평면의 그림으로 그리기에는 어려움이 있다.

<img src="{{site.image_url}}/study/3d_vector.png" style="width:25em; display: block; margin: 0px auto;">

벡터 간의 덧셈이 이뤄지는 과정은 다음과 같이 평행사변형 꼴로 표현된다.

$$
\begin{bmatrix}
1\\2
\end{bmatrix} \
+ \
\begin{bmatrix}
2\\0
\end{bmatrix}
= \
\begin{bmatrix}
3\\2
\end{bmatrix}
$$

<img src="{{site.image_url}}/study/2d_vector_add.png" style="width:22em; display: block; margin: 0px auto;">
