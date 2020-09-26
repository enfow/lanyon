---
layout: post
title: The Real Numbers
category_num : 1
---

# The Real Numbers

- Robert G. Bartle, Donal R. Sherbert의 Introduction to real analysis를 참고하여 작성했습니다.
- update at 2020.09.26

## Complete Ordered Field

**실수 체계(Real Number System)**는 **Complete Ordered Field** 이다. 다른 말로 하면 실수 체계는 다음 세 가지 특성을 가진다.

- **Algebraic Properties**: 체(Field)로서의 특성(Field Properties)을 가진다.
- **Order Properties**: 요소 간의 대소 비교가 가능하다.
- **Completeness Properies**: 완비성을 가진다.

위의 세 가지 특성은 **공리(Axiom)**이므로 증명이 불필요하다. 각각에 대해 하나씩 살펴보면 다음과 같다.

### Algebraic Properties

체(Field) $$(F, +, \cdot)$$란 두 Binary Operator, 덧셈($$+$$)과 곱셈($$\cdot$$)이 정의되어 있으면서 다음 10가지 특성을 가지는 대수적 체계(Algebraic System, Structure)를 말한다. 아래 10가지 특성은 **Field Axioms**라고도 한다.

$$
\eqalignno{
(A1) & \ a + b = b + a & \forall a, b \in \mathcal R \\
(A2) & \ (a + b) + c = a + (b + c) & \forall a, b, c \in \mathcal R \\
(A3) & \ a + 0 = a, \ 0 + a = 0 & 0 \in \mathcal R\\
(A4) & \ -a \in \mathcal R \qquad s.t. \ a + (-a) = 0, \ (-a) + a = 0 & \forall a \in \mathcal R\\
(M1) & \ a \cdot b = b \cdot a & \forall a, b \in \mathcal R \\
(M2) & \ (a \cdot b) \cdot c = a \cdot (b \cdot c) & \forall a, b, c \in \mathcal R \\
(M3) & \ a \cdot 1 = a, \ 1 \cdot a = a & 1 \in \mathcal R\\
(M4) & \ {1 \over a} \in \mathcal R \qquad s.t. a \cdot {1 \over a} = 1, \ {1 \over a} \cdot a = 1 & \forall a \in \mathcal R, \ a \neq 0\\
(D) & \ a \cdot (b + c) = (a \cdot b) + (a \cdot c), \ (b + c) \cdot a = (b \cdot a) + c (c \cdot a) & \forall a, b, c \in \mathcal R
}
$$

간단히 말해 덧셈과 곱셈에 대해 분배법칙, 교환법칙, 결합법칙이 성립하고, 덧셈의 항등원 $$0$$과 곱셈의 항등원 $$1$$이 정의되어 있는 대수적 체계를 체라고 하며, 실수 집합 $$\mathcal R$$은 위의 특성을 가지는 체의 대표적인 예가 된다. 참고로 뻴셈과 나눗샘은 각각 음수의 덧셈, 역수의 곱셈으로 표현이 가능하다는 점에서 쉽게 생각해 체는 사칙연산이 가능한 집합이라고도 볼 수 있다.

### Order Properties

Order Properties는 간단히 말해 요소 간의 대소성이 있다는 특성을 의미하며, 이를 위해서는 양수성(Prsitivity)와 부등성(Inequality)을 가져야 한다. 우선 양수성에 대해 먼저 살펴보면, $$\mathcal R$$의 공집합이 아닌 부분집합으로서 다음과 같은 특성을 가지는 양의 실수(Positive Real Number) 집합 $$\mathcal P$$가 존재한다는 것을 의미한다.

$$
\eqalign{
&(1) \ \text{if } a, b \text{ belong to } \mathcal P, \text{ then } a + b \text{ belongs to } \mathcal P\\
&(2) \ \text{if } a, b \text{ belong to } \mathcal P, \text{ then } ab \text{ belongs to } \mathcal P\\
&(3) \ \text{if } a \text{ belong to } \mathcal P, \text{ then exactly one of the following holds: }\\
& \qquad (i) \ a \in \mathcal P \qquad (ii) \ a = 0 \qquad (iii) \ -a \in \mathcal P
}
$$

여기서 세 번째 특성은 실수의 삼분위성(Trichotomy Property)으로 실수는 양수, 0, 음수로 나누어진다는 특성을 나타낸다. 이와 같이 양수성을 먼저 정의하는 이유는 이를 통해 실수의 각 요소 간 대소 비교가 가능하기 때문이다.

$$
\eqalign{
&\text{[Definition]}\\
& \qquad (a) \ \text{if } a - b \in \mathcal P, \text{then we write } a > b \text{ or } b < a\\
& \qquad (b) \ \text{if } a - b \in \mathcal P \cup \{ 0 \}, \text{then we write } a \geq b \text{ or } b \leq a\\
}
$$

실수의 삼분위성에 따라 모든 실수쌍 $$a, b$$ 는 서로 $$a < b$$, $$a = b$$ 또는 $$a > b$$ 중 하나의 관계를 가진다. 실수의 각 요소에 대한 대소의 의미를 정의했고, 이에 따라 아래 세 가지의 정리를 도출할 수 있다.

$$
\eqalign{
& \text{[Theorem]} \\
& \qquad (1) \text{If } a \in \mathcal R \text{ and } a \neq 0, \text{ then } a^2 > 0 \\
& \qquad (2) 1 > 0 \\
& \qquad (3) \text{If } n \in \mathcal N, \text{ then } n > 0
}
$$

각각에 대해 하나씩 살펴보면 $$(1)$$의 경우 $$a \neq 0$$이라고 했으므로, $$a$$는 $$a > 0$$ 또는 $$a < 0$$이라고 할 수 있다. 이때

- $$a > 0$$이면 $$a \cdot a > 0$$ 이므로 $$a^2 > 0$$을 만족하고
- $$a < 0$$이면 $$(-a) \cdot (-a) > 0$$ 이므로 이 또한 $$a^2 > 0$$을 만족하므로

$$a^2 > 0$$ 이라는 것을 증명할 수 있다. $$(2)$$는 $$(1)$$에서 $$a = 1$$ 이라 하면 $$1^2 = 1 > 0$$ 이라는 것을 쉽게 확인이 가능하다. 끝으로 $$1 > 0$$ 이라는 것을 증명했으므로 $$1$$에서부터 시작하여 계속 $$1$$을 더해나가는 집합인 자연수 $$\mathcal N$$ 또한 항상 $$0$$보다 크다는 것을 알 수 있다.

### Completeness Properties

**실수의 완비성(Completeness)** 또한 위의 두 가지 특성과 마찬가지로 증명이 필요없고, 다만 그 특성이 존재함을 확인할 수 있을 뿐이다. 실수의 완비성을 보여주는 대표적인 방법은 **상한(Supremum)**의 개념을 사용하는 것이다.

#### Upper Bound & Supremum

상한이 무엇인지 알기 위해서는 Boundary에 대해 먼저 알아야 한다. 실수에서 Boundary는 **상계(Upper bound)**와 **하계(Lower Bound)** 두 가지가 존재하며 공집합이 아닌 부분집합 $$S \subset \mathcal R$$에 대해 다음과 같이 정의된다.

- 어떤 $$u \in \mathcal R$$에 대해 $$S$$의 모든 요소가 $$u$$보다 작거나 같다면($$s \leq u$$) 부분집합 $$S$$는 상계를 가진다(Bounded Above)고 하고, $$u$$를 $$S$$의 상계라고 한다.
- 어떤 $$u \in \mathcal R$$에 대해 $$S$$의 모든 요소가 $$u$$보다 크거나 같다면($$s \geq u$$) 부분집합 $$S$$는 하계를 가진다(Bounded Below)고 하고, $$u$$를 $$S$$의 하계라고 한다.

이때 **상한(Supremum)**은 $$S$$가 상계를 가지고 있을 때 다음 두 가지 조건을 만족하는 실수 $$u \in \mathcal R$$로 정의된다. **하계(Infimum)**는 반대로 정의되므로 생략한다.

- $$u$$는 $$S$$의 상계이다.
- 어떤 $$S$$의 상계 $$v$$에 대해 $$u \leq v$$가 성립한다.

기호로 상계는 $$\text{sup } S$$, 하계는 $$\text{inf } S$$로 표기한다.

<img src="{{site.image_url}}/study/real_number_sup_and_inf.png" style="width:30em; display: block; margin: 0px auto;">

모든 실수의 부분집합은 아래와 같이 네 가지 경우로 나누어 볼 수 있다.

- 상한과 하한을 모두 가지는 경우
- 상한만 가지는 경우
- 하한만 가지는 경우
- 상한과 하한을 모두 가지지 않는 경우

상한의 개념과 관련하여 확인할 수 있는 실수의 완비성은 상계를 가지는 모든 공집합이 아닌 실수의 부분집합이 실수인 상한을 가진다는 것이다. 이러한 점에서 실수의 완비성을 실수의 상한성(Supremum Property of $$\mathcal R$$)이라고 하기도 한다.

#### Boundary of Functions

함수에 대해서도 Boundary를 논할 수 있다. 어떤 함수 $$f : D \rightarrow \mathcal R$$에 대해 치역 $$f(D) = \{ f(x) : x \in D \}$$가 상계를 가지면 함수 $$f$$가 상계를 가진다고 한다. 그리고 함수 $$f$$가 상계와 하계를 모두 가지면 "$$f$$ is Bounded"라고 표현한다.

- 모든 $$x \in D$$에 대해 항상 $$f(x) \leq g(x)$$가 성립하면 $$\text{sup } f(D) \leq \text{sup } g(D)$$라고 한다. $$\text{sup }_{x \in D} f(x) \leq \text{sup }_{x \in D} g(x)$$로 표기하기도 한다.
- 모든 $$x, y \in D$$에 대해 항상 $$f(x) \leq g(y)$$가 성립하면 $$\text{sup } f(D) \leq \text{inf } g(D)$$라고 한다. $$\text{sup }_{x \in D} f(x) \leq \text{inf }_{y \in D} g(y)$$로 표기하기도 한다.

## Density Theorem

실수는 **유리수(Rational Number)**와 **무리수(Irrational Number)** 두 가지로 나누어진다. 이때 유리수는 셀 수 있는 무한집합이고, 무리수는 셀 수 없는 무한 집합이므로 무리수가 유리수보다 더 많다. 하지만 실수의 부분집합인 유리수 집합은 실수 집합에 대해 조밀(dense)하다고 표현한다. 이때 조밀하다는 것은 다음을 의미한다.

$$
\eqalign{
&\text{[Theorem]}\\
& \qquad \text{If } x \text{ and } y \text{ are any real numbers with } x < y, \\
& \qquad \text{then there exists a rational number } r \in \mathcal Q \text{ such that } x < r < y.
}
$$
