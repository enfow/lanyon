---
layout: post
title: Approximation by Superposition of a Sigmoidal Function
category_num : 1
keyword: '[Universal Approximation]'
---

# 논문 제목 : Approximation by Superposition of a Sigmoidal Function

- G. Cybenko
- 1989
- [논문 링크](<https://link.springer.com/article/10.1007/BF02551274>)
- 2020.09.23 정리

## Summary

- Universal Approximation Theorem은 하나의 뉴럴 네트워크 레이어와 비선형 활성 함수만으로 가능한 모든 연속 함수를 근사할 수 있다는 것을 수학적으로 보이고 있다.
- 이때 비선형 활성 함수 $$\theta$$는 **Continuous Discriminatory Function** 이어야 한다. 대표적으로 Sigmodal한 특성을 가지는 함수들이 있다.

## Artificial Neural Network

어떤 $$n$$차원의 입력 값 $$x \in \mathcal R^n$$에 대한 **인공 신경망(Artificial Neural Network)**은 다음 수식과 같다.

$$
\Sigma_{j=1}^N \alpha_j \sigma(y_j^T x + \theta_j)
$$

하나의 뉴럴 네트워크 레이어를 통과한다는 것은 위의 수식에서 확인할 수 있듯 Input $$x$$와 Weight $$y_j \in \mathcal R^n$$의 내적에 Bias $$\theta_j$$를 더하고 그것을 활성 함수에 통과시키는 작업을 해당 레이어 출력의 차원 수 $$j$$만큼 반복하는 것이 된다. 이러한 점에서 인공신경망은 뉴럴 네트워크 레이어와 비선형 활성 함수(Non-linear Activation Function)들이 중첩되어 있는 것이라고 할 수 있다.

여기서 당연히 다음과 같은 의문이 들 수 있다.

- 뉴럴 네트워크 레이어와 비선형 함수를 중첩하는 것만으로도 원하는 출력 값을 얻을 수 있는가
- 원하는 네트워크 출력을 얻기 위해 뉴럴 네트워크 레이어와 비선형 활성 함수를 얼마나 쌓아야 하는가

논문에서는 **하나의 뉴럴 네트워크 레이어와 임의의 연속적인 시그모달 비선형 활성 함수(Arbitrary Continuous Sigmoidal Nonlinearity)**만으로도 충분히 원하는 출력 값을 얻을 수 있음을 보여준다. 이때 충분하다는 것은 정확하게 표현하는 것이 아닌 근사(Approximation)에 만족한다는 것으로 이해할 수 있다.

## Cybenko's theorem

$$n$$차원의 Unit Cube $$[0,1]^n$$을 $$I_n$$, 그리고 $$I_n$$ 상에서 정의되는 연속 함수의 공간을 $$C(I_n)$$이라고 하자. 이때 우리의 목표는 다음과 같이 $$G(x)$$로 정의되는 하나의 뉴럴 네트워크 레이어와 시그모이드 활성 함수만으로 $$C(I_n)$$에 대해 dense 하다는 것을 보이는 것이다. 참고로 어떤 위상 공간 $$X$$의 Subset $$A$$가 있다고 할 때, $$X$$의 모든 point $$x$$가 $$A$$에 속하거나 $$A$$의 극한점일 때 $$A$$는 $$X$$에 대해 dense 하다고 표현한다.

$$
G(x) = \Sigma_{j=1}^N \alpha_j \sigma(y_j^T x + \theta_j)
$$

쉽게 말해 dense 하다는 것은 $$G(x)$$로 $$I_n$$에서 정의될 수 있는 모든 Continuous Function을 커버할 수 있다는 것을 의미한다. **Cybenko's theorem**은 이것이 성립한다는 것에 관한 증명이다.

### Definitions

구체적인 증명에 앞서 논문에서는 Disciminatory와 Sigmodal이 무엇인지 다음과 같이 정의한다.

#### Disciminatory

Unit Cube $$I_n$$에 대해 유한하고, 부호가 있는 Regular Borel Measure들의 집합을 $$M(I_n)$$이라고 하자. 어떤 Measure $$\mu \in M(I_n)$$가 있을 때 모든 $$y \in \mathcal R^n$$과 $$\theta \in \mathcal R$$에 대해 아래와 같은 공식을 만족하면 $$\mu = 0$$이 성립하는 경우를 두고 $$\sigma$$가 **Discriminatory** 하다고 한다.

$$
\int_{I_n} \sigma(y^T x + \theta) d \mu (x) = 0
$$

#### Sigmoidal

$$\sigma$$가 **Sigmoidal**하다는 것은 다음과 같이 정의될 때를 의미한다.

$$
\sigma (t) \rightarrow
\cases{
1 \text{ as } t \rightarrow +\infty \cr
0 \text{ as } t \rightarrow -\infty
}
$$

### Theorem

$$
\eqalign{
&\text{Let } \sigma \text{ be any continuous discriminatory function. Then finite sums of the form }\\ \\
&\qquad G(x) = \Sigma_{j=1}^N \alpha_j \sigma(y_j^T x + \theta_j) \\ \\
&\text{are dense in } C(I_n). \text{ In other words, given any } f \in C(I_n) \text{ and } \epsilon > 0, \text{ there is a sum, } \\ \\
& G(x), \text{of the above form, for which }\\ \\
&\qquad \lvert G(x) - f(x) \rvert < \epsilon \qquad \text{ for all } x \in I_n
}
$$

Theorem은 $$I_n$$상에서 $$G(x)$$를 통해 표현할 수 있는 함수의 집합을 $$S \subset C(I_n)$$이라고 한다면, $$S$$의 closure가 $$I_n$$에서 정의되는 전체 연속함수 집합 $$C(I_n)$$이 된다는 것으로도 이해할 수 있다. 여기서 Closure(폐포)란 주어진 위상공간의 부분집합을 포함하는 가장 작은 닫힌 집합을 의미하므로 $$C(I_n)$$의 부분집합인 $$S$$의 Closure가 $$C(I_n)$$라면 $$S$$로 전체 $$C(I_n)$$을 커버할 수 있다는 것을 의미한다.

### Proof

Cybenko's theorem은 귀류법을 통해 증명이 이뤄진다. 따라서 증명은 $$S$$의 closure가 전체 $$C(I_n)$$이 된다는 것을 부정하는 것으로 시작한다. 이를 위해 $$S$$의 Closure를 $$C(I_n)$$의 부분집합인 $$R$$로 가정한다.

증명에는 두 가지 부가적인 Theorem을 사용하게 되는데, 첫 번째는 **Hahn-Banach Theorem**으로, $$C(I_n)$$에 있어 $$L(S) = L(R) = 0, \ L \neq 0$$을 만족하는 Bounded Linear Functional $$L$$이 존재한다는 것을 알 수 있다. 두 번째로 사용되는 Theorem은 **Riesz Representation Theorem**이다. 이에 따르면 $$C(I_n)$$의 모든 원소 $$h$$에 대해 $$L$$이 다음과 같이 정의될 수 있다. 이때 $$\mu$$는 Disciminatory의 정의에서 사용되는 어떤 Regular Borel Measure 이다.

$$
L(h) = \int_{I_n} h(x) d \mu (x)
$$

여기서 모든 $$y$$와 $$\theta$$에 대해 $$\sigma(y^T x + \theta)$$는 $$R$$의 원소이므로 다음이 성립힌다.

$$
L(h) = \int_{I_n} \sigma(y^T x + \theta) d \mu (x) = 0
$$

Theorem에서 $$\sigma$$를 Continuous Discriminatory Function으로 정의하였으므로 이 경우 Discriminatory의 정의에 따라 $$\mu = 0$$를 만족함을 알 수 있다. 이는 $$R$$이 곧 $$C(I_n)$$이 된다는 것을 의미한다. 따라서 $$S$$의 closure가 전체 $$C(I_n)$$이 되지 못한다는 명제는 틀렸다.

### General Form of Artificial Neural Network

$$
G(x) = \Sigma_{j=1}^N \alpha_j \sigma(y_j^T x + \theta_j)
$$

$$\sigma(y^T x + \theta)$$가 $$C(I_n)$$에 대해 dense 하기 때문에 $$\sigma$$가 Continuous Discriminatory Function 이기만 하면 위와 같이 일반적인 형태의 인공신경망 $$G(x)$$ 또한 $$C(I_n)$$에 대해 dense 하다고 할 수 있다.
