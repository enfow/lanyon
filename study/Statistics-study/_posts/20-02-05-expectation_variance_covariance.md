---
layout: post
title: Expectation, Variance and Covariance
category_num: 1
---

# Expectation, Variance and Covariance

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다. 
- update date : 2020.02.05

## 1. Expectation

$$x$$가 어떤 확률 분포 $$P(x)$$를 따를 때, 어떤 함수 $$f(x)$$의 평균(average or mean value)을 **기대값**(expecation)이라고 한다. 수학적으로는 다음과 같이 표현할 수 있다.

$$
E_{x \backsim P}[f(x)] = \Sigma_x P(x)f(x)
$$

$$
E_{x \backsim P}[f(x)] = \int_x P(x)f(x)
$$

이때 $$x$$의 확률 분포가 자명한 경우에는 $$x \backsim P$$를 생략하고 $$x$$만 쓰거나, 아예 쓰지 않는 경우도 많다.

$$
E_{x \backsim P}[f(x)] = E_{x}[f(x)] = E[f(x)]
$$

## 2. Variance and Standard deviation

$$f(x)$$ **분산**(variance)은 $$f(x)$$의 값이 얼마나 퍼져있는지(vary)를 측정하는 척도라고 할 수 있으며, 기대값과의 차이를 통해 계산한다. 분산은 기본적으로 제곱을 하여 계산하기 때문에 확률변수 단위의 영향을 받는다. 이러한 문제를 해결하기 위해, 즉 단위의 문제를 맞추기 위해 분산에 제곱근을 씌운 **표준편차**(standard deviation, std)가 빈번하게 사용된다. 참고로 standard deviation에서 deviation이란 편차를 뜻하며, 이는 개별 데이터와 평균의 차이를 의미한다.

$$
\eqalign{
&Var(f(x)) = E[(f(x) - E[f(x)])^2] \\
&Std(f(x)) = \root \of {Var(f(x))}
}
$$

즉, 분산은 $$f(x)$$의 평균에서 떨어진 거리의 제곱의 평균과 같다.

### Characteristics of Variance

#### 1) Independent with Expectation

분산은 기대값과 독립이기 때문에 전체 $$f(x)$$의 값에 $$b$$가 더해져 기대값이 $$b$$만큼 증가했다 하더라도 분산은 동일하다.

$$
Var(f(x) + b) = Var(f(x))
$$

하지만 전체 값에 어떤 수를 곱하게 되면 분산은 그 제곱만큼 커지게 된다. 따라서 아래와 같이 정리할 수 있다.

$$
Var(af(x) + b) = a^2 Var(f(x))
$$

#### 2) The Sum of two independent variables' Variance is equal to the sum of each Variance.

보다 구체적으로 아래와 같은 공식이 성립한다.

$$
Var(af(x) + bg(y)) = a^2 Var(f(x)) + b^2Var(g(y)) + 2abCov(f(x), g(y))
$$

여기서 $$Cov(f(x), g(y))$$는 공분산을 의미하며, 두 변수가 독립인 경우에는 0이기 때문에 결과적으로 두 독립 변수의 합의 분산은 각 독립 변수의 분산의 합과 동일하다.

### Standardization

표준화는 두 개 이상의 분포 간의 관계를 확인할 때 단위의 영향을 없애기 위한 방법이다. 아래의 공식을 이용하면 전체 데이터의 분산을 1로 맞출 수 있다.

$$
{f(x) - E[f(x)] \over Std(f(x))}
$$

이를 python 코드를 통해 확인하면 다음과 같다.

```
>>> import numpy as np
>>> array = np.array([1,2,3,4,5])

# 분산과 표준편차
>>> np.var(array)
2.0
>>> np.std(array)
1.4142135623730951

#표준화
>>> standard = (array - np.mean(array)) / np.std(array)

#표준화의 결과 - 표준편차가 1이 된다
>>> standard
array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
>>> np.std(standard)
0.9999999999999999
```

### Correlation

[위키](<https://ko.wikipedia.org/wiki/%EC%83%81%EA%B4%80_%EB%B6%84%EC%84%9D>)에서는 상관관계를 다음과 같이 정의하고 있다.

- 두 변수는 서로 독립적인 관계이거나 상관된 관계일 수 있으며 이때 두 변수간의 관계의 강도를 상관관계(Correlation, Correlation coefficient)라 한다. 상관분석에서는 상관관계의 정도를 나타내는 단위로 모상관계수로 ρ를 사용하며 표본 상관 계수로 r 을 사용한다.

즉, 상관관계란 두 변수가 얼마나 관련이 있는지에 관한 것이며, 이를 수치적으로 표현한 것을 **상관관계**라고 한다. 상관관계를 확인하는 방법은 여러가지가 있지만 아래의 피어슨 상관 계수를 가장 많이 사용한다.

$$
r = { {\Sigma_i^n(X_i-\mu_1)(Y_i - \mu_2) \over n-1 } \over \root \of {\Sigma_i^n (X_i - \mu_1)^2 \over n-1} \root \of {\Sigma_i^n (Y_i - \mu_1)^2 \over n-1}}
$$

참고로 모집단을 가정하는 모상관계수의 경우에는 기호로 $$\rho$$를 사용하며 $$n-1$$이 아닌 $$n$$으로 나누어 준다고 한다.

## 3. Covariance and Correlation

### Covariance

**공분산**(covariance)은 두 확률 변수 간의 선형적인 상관성을 측정하기 위해 사용된다. 수학적인 정의는 아래와 같다.

$$
Cov(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]
$$

즉, 분산은 동일한 확률 변수 간의 공분산이라고 할 수 있다.

### Covariance and Correlation

공분산의 절대값의 크기가 크면 클수록 두 변수 간의 상관성이 높다고 할 수 있다.

- $$Cov(X, Y)$$ > 0 : 양의 상관관계
- $$Cov(X, Y)$$ < 0 : 음의 상관관계
- $$Cov(X, Y)$$ = 0 : 상관관계 없음

이는 위에서 정의한 상관계수와도 관련이 깊다. 다시 말해 상관계수 식을 조금 더 전개하면 공분산 식이 도출된다.

$$
\eqalign{
r &= { {\Sigma_i^n(X_i-\mu_1)(Y_i - \mu_2) \over n-1 } \over \root \of {\Sigma_i^n (X_i - \mu_1)^2 \over n-1} \root \of {\Sigma_i^n (Y_i - \mu_1)^2 \over n-1}} \\
&= { {\Sigma_i^n(X_i-\mu_1)(Y_i - \mu_2)} \over \root \of {\Sigma_i^n (X_i - \mu_1)^2} \root \of {\Sigma_i^n (Y_i - \mu_1)^2}}\\
&= {Cov(X,Y) \over \sigma_1 \sigma_2}
}
$$

역으로 생각해보면 공분산을 통해 상관계수를 쉽게 계산할 수 있다.

### Characteristics of Covariance

#### 1) Covariance and Linear Correlation

공분산이 0이라고 해서 두 변수가 독립이라고는 할 수 없다. 공분산은 선형 상관관계에만 영향을 받기 때문에 공분산이 0이라는 것은 선형 독립은 의미하지만 비선형 독립을 보장하지는 못하기 때문이다. 이러한 점은 두 변수 간의 상관관계를 측정하는 지표로서 공분산의 한계라고 할 수 있다.

#### 2) Formula

- $$Cov(X,X) = Var(X)$$.
- $$Cov(X,Y) = Cov(Y,X)$$.
- $$Cov(aX,bY) = ab Cov(X,Y)$$.

#### 3) Dot Product and Covariance

벡터 $$\boldsymbol{a}, \boldsymbol{b}$$의 내적은 다음과 같이 정의된다.

- 기하학적 정의

$$
\boldsymbol{a} \cdot \boldsymbol{b} = \lvert \boldsymbol{a} \rvert \cos \theta \lvert \boldsymbol{b} \rvert
$$

- 수학적 정의

$$
\boldsymbol{a} \cdot \boldsymbol{b} = \Sigma a_i b_i
$$

공분산은 이러한 벡터 간의 내적을 통해서도 구할 수 있다.

$$
\eqalign{
    Cov(X,Y) &= E[(X-\mu_1)(Y-\mu_2)]\\
    &=E[XY-X\mu_2-Y\mu_1+\mu_1\mu_2]\\
    &=E[XY]-E[X]\mu_2-E[Y]\mu_1+\mu_1\mu_2\\
    &=E[XY]-\mu_1\mu_2 - \mu_2\mu_1+\mu_1\mu_2\\
    &=E[XY]-\mu_1\mu_2\\
    &={1 \over (n-1)} \Sigma X_iY_i - \mu_1\mu_2
}
$$

위의 식에서 $$\Sigma X_iY_i$$를 벡터 X와 벡터 Y 간의 내적이다. 즉 공분산은 두 벡터의 내적에 $$n-1$$을 나누고 평균의 곱을 뺀 것으로 구할 수 있다. 이러한 특성은 공분산 행렬을 구할 때 많이 사용한다.

#### 4) Covariance Matrix and Product

$$nXm$$의 데이터가 있고, row의 수 $$n$$은 데이터의 개수를 column의 수 $$m$$은 feature의 수를 의미한다고 할 때, 내적과 공분산의 관계를 이용하면 두 벡터의 곱으로 공분산 행렬을 구할 수 있다.

이때 $$\mu_1\mu_2 = 0$$으로 만들기 위해서 두 분포의 평균을 0으로 가정하고 전개한다.

$$
\eqalign{
{1 \over (n-1)}X^T X &=
\begin{bmatrix}
{1 \over (n-1)} X_1 \cdot X_1 && {1 \over (n-1)} X_1 \cdot X_2 && {1 \over (n-1)} X_1 \cdot X_3 && ... \\ 
{1 \over (n-1)} X_2 \cdot X_1 && {1 \over (n-1)} X_2 \cdot X_2 && {1 \over (n-1)} X_2 \cdot X_3 && ... \\ 
{1 \over (n-1)} X_3 \cdot X_1 && {1 \over (n-1)} X_3 \cdot X_2 && {1 \over (n-1)} X_3 \cdot X_3 && ... \\ 
... && ... && ...
\end{bmatrix} \\
&=
\begin{bmatrix}
Var(X_1) && Cov(X_1, X_2) && Cov(X_1, X_3) && ... \\ 
Cov(X_2, X_1) && Var(X_2) && Cov(X_2, X_3) && ... \\ 
Cov(X_3, X_1) && Cov(X_3, X_2) && Var(X_3) && ... \\ 
... && ... && ...
\end{bmatrix} \\
}
$$