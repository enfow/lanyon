---
layout: post
title: Expectation, Variance and Covariance
category_num: 1
---

# Expectation, Variance and Covariance

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다. 
- update date : 2020.02.05

## 1. Expectation

### 기대값

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

### 분산

$$f(x)$$ **분산**(variance)은 $$f(x)$$의 값이 얼마나 퍼져있는지(vary)를 측정하는 척도라고 할 수 있으며, 기대값과의 차이를 통해 계산한다. 분산은 기본적으로 제곱을 하여 계산하기 때문에 확률변수의 단위와 차이가 있으며, 이러한 문제를 해결하기 위해, 즉 단위의 문제를 맞추기 위해 분산에 제곱근을 씌운 **표준편차**(standard deviation, std)가 빈번하게 사용된다.

$$
Var(f(x)) = E[(f(x) - E[f(x)])^2]
$$

즉, 분산은 $$f(x)$$의 평균에서 떨어진 거리의 제곱의 평균과 같다.

### 분산의 성질

#### 1) 분산은 기대값에 대해 독립이다.

따라서 전체 $$f(x)$$의 값에 $$b$$가 더해져 선형 이동하더라도 분산은 동일하다.

$$
Var(f(x) + b) = Var(f(x))
$$

하지만 전체 값에 어떤 수를 곱하게 되면 분산은 그 제곱만큼 커지게 된다. 따라서 아래와 같이 정리할 수 있다.

$$
Var(af(x) + b) = a^2 Var(f(x))
$$

#### 2) 두 개의 독립 확률변수의 합의 분산은 각각의 분산의 합과 같다.

보다 구체적으로 아래와 같은 공식이 성립한다.

$$
Var(af(x) + bg(y)) = a^2 Var(f(x)) + b^2Var(g(y)) + 2abCov(f(x), g(y))
$$

여기서 $$Cov(f(x), g(y))$$는 공분산을 의미하며, 두 변수가 독립인 경우에는 0이 된다.

## 3. Covariance and Correlation

### 공분산

**공분산**(covariance)은 두 확률 변수 간의 선형적인 상관성을 측정하기 위해 사용된다. 수학적인 정의는 아래와 같다.

$$
Cov(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]
$$

공분산의 절대값의 크기가 크면 클수록 두 변수 간의 상관성이 높다고 할 수 있다.

- $$Cov(X, Y)$$ > 0 : 양의 상관관계
- $$Cov(X, Y)$$ < 0 : 음의 상관관계
- $$Cov(X, Y)$$ = 0 : 상관관계 없음

### 모상관계수

공분산은 양수, 음수 여부에 따라 어떤 유형의 상관관계가 있는지 확인할 수는 있지만, 두 상관 변수의 스케일에 영향을 받으므로 얼마나 강한 상관관계를 가지는지 확인하기에는 부적절하다. **모상관계수(correlation)**는 스케일의 영향을 축소시켜 상관관계의 정도를 나타내는 단위로 사용된다.

$$
\rho = {Cov(f(x), g(y)) \over \root \of {Var(f(x)) Var(g(y))}}
$$

### 공분산의 성질

#### 1) 공분산은 선형 상관성과 관련있다.

공분산이 0이라고 해서 두 변수가 독립이라고는 할 수 없다. 공분산은 선형 상관관계에만 영향을 받기 때문에 공분산이 0이라는 것은 선형 독립은 의미하지만 비선형 독립을 보장하지는 못하기 때문이다. 이러한 점은 두 변수 간의 상관관계를 측정하는 지표로서 공분산의 한계라고 할 수 있다.

#### 2) 공분산과 관련된 공식

- $$Cov(X,X) = Var(X)$$.
- $$Cov(X,Y) = Cov(Y,X)$$.
- $$Cov(aX,bY) = ab Cov(X,Y)$$.
