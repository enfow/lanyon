---
layout: post
title: Shannon Entropy
category_num: 5
---

# Shannon Entropy

- Christopher M. Bishop의 Pattern Recognition and Machine Learning과 Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다.
- update date : 2020.02.06, 2020.02.21

## Information Theory

정보이론은 어떤 signal이 주어졌을 때 그에 담긴 정보의 양이 얼마나 되는지에 대해 관심을 갖는 응용 수학 분야이다. 정보이론은 자주 발생하지 않는 특수한 사건이 벌어졌다는 것을 알았을 때 빈번하게 발생하는 사건보다 더 많이 놀라게 된다는 기본적인 아이디어에서 출발한다. 그리고 이를 **엔트로피**(entropy)라는 개념으로 다루게 된다.

## Entropy

Elements of information theory(Cover&Thomas)에서는 엔트로피를 다음과 같이 정의한다.

- measure of the uncertainty of a random variable

어떤 확률 변수의 불확실성의 정도를, 즉 얼마나 무작위적인가를 수치화한 것이라고 할 수 있다. 어떤 확률 변수의 불확실성의 정도를 확인하기 위해서는 우선 개별 사건들이 가지는 정보량을 파악해야 한다. 이를 Information Gain이라고 한다.

### Information Gain

기본적으로 정보이론은 다음 세 가지 가정을 바탕으로 한다.

1. 발생 빈도가 높은 사건이 관찰되었다는 것은 적은 정보를 가지고 있다.
2. 발생 빈도가 낮은 사건이 관찰되었다는 것은 많은 정보를 가지고 있다.
3. 독립 사건은 추가적인 정보가 된다.

이때 어떤 특정한 사건 $$x$$가 가지는 Information Gain을 평가하는 함수 $$h(x)$$가 있다고 하자. 위의 첫 번째, 두 번째 가정에 따르면 즉 $$h(x)$$는 사건의 발생 빈도, 즉 확률 $$p(x)$$에 대한 단조 함수(monotonic function)이다. 그리고 확률 값이 높으면 Information Gain은 적다는 점에서 다음과 같은 특성을 띈다.

$$
\text{if} \ p(x) > p(y) \qquad \text{then} \ h(x) < h(y)
$$

두 개의 독립적인 사건이 발생했다고 하면 정보량은 각 사건의 정보량을 합한 것이라는 점은 다음과 같이 표현이 가능하다.

$$
h(x, y) = h(x) + h(y)
$$

이러한 점을 고려할 때 $$h(x)$$를 다음과 같이 표현할 수 있다.

$$
h(x) = - \log p(x)
$$

이때 로그의 밑을 자연상수 $$e$$ 또는 2로 한다.

이와 같이 확률 값에 음의 로그를 취하게 되면 확률이 높으면 높을수록 낮은 값을, 낮으면 낮을수록 높은 값을 반환하게 되므로 위의 가정에 부합한다고 할 수 있다.

### Shannon Entropy

위에서 정의한 Information Gain은 말 그대로 어떤 사건이 발생했다는 정보가 가지는 정보량이다. 단일 사건이 갖는 정보량을 이용하면 어떤 분포를 따르는 확률 변수 $$X$$의 정보량을 구할 수 있다.

$$
\eqalign{
H(X) &= E_{X \backsim P}[h(x)]\\ &= - E_{X \backsim P}[ \log P(x) ]\\ &= - \Sigma_x P(x) \log P(x)
}
$$

위의 식은 확률 변수 $$X$$가 따르는 확률 분포 $$P$$의 정보량이라는 뜻에서 $$H(P)$$로 표기하기도 하며, 어떤 분포에 따라 사건이 발생했다는 것을 알았을 때 기대되는 정보량으로 정의된다. 이와 같이 어떤 확률 분포가 가지는 정보량을 **Shannon Entropy**라고 한다.

### Characteristics of Shannon Entropy

#### 1) Uniform Sampling has Maximum Entropy

확률 값 $$p(x)$$는 $$0 \leqq p(x) \leqq 1$$의 범위를 가진다. 일단 음수가 될 수 없기 때문에 어떤 확률 값에 대해 $$p(x_i) = 1$$가 성립하면 다른 확률 값들은 모두 0이 된다. 이러한 점을 고려할 때 $$p(x)$$ 분포가 뾰족하게 튀어나와 있는 것보다 편평하게 퍼져 있는 형상을 띌 때 엔트로피가 보다 높다. 그리고 동일한 확률 공간에서 **엔트로피가 극대화되는 분포는 확률 변수가 균등 분포를 따를 때**라고 할 수 있다.

이는 Lagrange multiplication에 따라 확인할 수 있다. 아래의 경우 $$g_i(x,y) = c_i$$의 제약식을 가정한다.

$$
Lagrange \ : \  L(x,y,\lambda_1, \lambda_2, ..., \lambda_i) = f(x,y) - \Sigma_{i=1}^{N} \lambda_i(g_i(x,y)-c_i)
$$

확률 분포의 경우 모든 가능한 사건의 확률을 모두 더하면 1이라는 제약이 있다. 따라서 $$H$$의 극대화 식은 다음과 같이 표현할 수 있다.

$$
H = - \Sigma_i p(x_i) \log p(x_i) + \lambda ( \Sigma_i p(x_i) - 1 )
$$

위 식은 $$p(x_i) = 1 / M$$이 성립할 때 극대화 되며, 이때의 정확한 엔트로피 값 $$H(P) = \log M$$이 성립한다. $$M$$은 $$x_i$$의 개수, bin의 개수라고 할 수 있다.

#### 2) Lower Bound on the Number of Bits

Shannon은 문자열을 0,1과 같은 비트로 인코딩하는 방법을 연구하는 과정에서 엔트로피를 정의했다. Shannon의 논문 *noiseless coding theorem(1948)*에서는 엔트로피를 다음과 같이 정의한다.

- the entropy is a lower bound on the number of bits needed to transmit the state of a random variable

즉, 엔트로피를 **어떤 문자들(random variable)을 전송할 때 최소한의 비트 수**로 본다. 이는 다음과 같이 서로 다른 비율을 갖는 문자 $$a,b,c,d,e,f$$를 전송하는 문제를 통해 확인할 수 있다. 참고로 이때 $$\log$$의 밑은 2이다.

$$
\eqalign{
&S = \{ a,b,c,d,e,f \} \\
&p(a) = {1 \over 4} \quad
p(b) = {1 \over 4} \quad
&p(c) = {1 \over 8} \quad
p(d) = {1 \over 8} \quad
p(e) = {1 \over 8} \quad
p(f) = {1 \over 8} \\
}
$$

##### (a) Information Gain에 따른 encoding

이 경우 각각의 Information Gain은 다음과 같다.

$$
\eqalign{
&h(a) = 2 \quad
h(b) = 2 \quad
&h(c) = 3 \quad
h(d) = 3 \quad
h(e) = 3 \quad
h(f) = 3 \\
}
$$

a는 2비트로, c는 3비트 등으로 인코딩하면 최적이라는 것을 의미한다. 이에 따라

$$
H = {2 \over 4} \cdot 2 + {4 \over 8} \cdot 3 = 2.5
$$

엔트로피$$H$$는 2.5가 된다.

##### (b) random distirubtion을 가정한 encoding

모든 문자가 동일한 확률로 있다고 가정하면 각각의 문자의 Information Gain은 $$log_2 6$$이 된다. 이를 통해 엔트로피 $$H$$를 구하게 되면

$$
H = {2 \over 4} \cdot \log_2 6 + {4 \over 8} \cdot \log_2 6 = \log_2 6
$$

으로, $$\log_2 6 = 2.5849$$가 된다. 이는 (a)보다 큰 값이다.

##### (c) 잘못된 방법의 encoding

사용자가 실수로

$$
\eqalign{
&h(a) = 3 \quad
h(b) = 3 \quad
&h(c) = 2 \quad
h(d) = 2 \quad
h(e) = 3 \quad
h(f) = 3 \\
}
$$

로 인코딩을 진행했다고 가정하자. 즉 자주 나타나는 a,b에 큰 비트 수를 부여하고, 다른 c, d에 작은 비트 수를 부여한 경우이다. 이 경우 엔트로피는

$$
{2 \over 4} \cdot 3 + {2 \over 8} \cdot 2 + {2 \over 8} \cdot 3 = 2.75
$$

2.75로 가장 높은 것을 알 수 있다.

## Entropy of Continuous Variable: Differential Entropy

지금까지는 Discrete Distribution을 가정했었다. 하지만 이를 Continuous Distribution에 곧바로 적용하는 것에는 문제가 있다. 왜냐하면 Continuous Distribution에서는 개별 사건의 발생 확률이 0이기 때문이다. 이를 위해 여러 개의 bin으로 나누어 Discrete Distribution과 유사하게 접근하는 방법이 있다. bin의 크기를 $$\Delta$$로 가정하면 Continuous Distribution 또한 다음과 같이 표현이 가능하다.

$$
\int_{i\Delta}^{(i+1)\Delta} p(x) dx = p(x_i)\Delta
$$

이를 Discrete Distribution의 entropy를 구하는 수식에 적용하면 다음과 같이 수식을 전개할 수 있다.

$$
\eqalign{
&\Sigma_i p(x_i)\Delta \log p(x_i)\Delta\\
&=\Sigma_i p(x_i)\Delta \log p(x_i) + \Sigma_i p(x_i)\Delta \log \Delta \\
&=\Sigma_i p(x_i)\Delta \log p(x_i) + \log \Delta \qquad  \qquad (\because \Sigma_i p(x_i)\Delta = 0) \\
&=\lim_{\Delta \rightarrow 0} \Sigma_i p(x_i)\Delta \log p(x_i) + \lim_{\Delta \rightarrow 0} \log \Delta
}
$$

Continuous Random Variable을 가정하였기 때문에 수식 마지막 줄에서  $$\Delta$$를 0에 수렴하도록 하고 있다. 하지만 이 경우 문제가 발생하는데 두 번째 항 $$\lim_{\Delta \rightarrow 0} \log \Delta$$이 무한으로 발산한다는 점이다. 이러한 점 때문에 Continuous Distribution에서는 두 번째 항을 생략하고 첫 번째 항으로만 엔트로피를 구하게 되는데, 이를 **Differential Entropy**라고 한다.

$$
\eqalign{
H(x)
&= \lim_{\Delta \rightarrow 0} \Sigma_i p(x_i)\Delta \log p(x_i) \\
&= \int p(x) \log p(x) dx
}
$$

### Entropy of Gaussian distribution

평균과 분산 $$(\mu, \sigma)$$이 정해져 있을 때, entropy가 가장 큰 분포는 가우시안 분포 $$N(\mu, \sigma^2)$$이다. 이때의 엔트로피는 다음과 같다.

$$
H(X) = {1 \over 2} (1 + \ln (2 \pi \sigma^2))
$$

## Kullback Leibler Divergence

**쿨백 라이블러 발산**(Kullback Leibler Divergence, KLD)은 정보량 개념을 이용하여 분포 간의 차이를 계산하는 함수다. 동일한 확률 변수 $$X$$에 대한 다른 분포 $$P(X)$$, $$Q(X)$$ 간의 KLD는 다음과 같이 정의된다.

$$
\eqalign{
D_{KL}(P \| Q) &= E_{X \backsim P} [\log { P(X) \over Q(X) }]\\
&= \Sigma_x ( P(x) \log{P(x) \over Q(x)} )\\
&= \Sigma_x ( P(x)\log{P(x) - P(x) \log Q(x)})\\
}
$$

이러한 쿨백 라이블러 발산은 다음과 같은 특성을 갖는다.

- 확률 분포 $$P(X)$$와 $$Q(X)$$가 동일하면 $$D_{KL}(P \| Q) = 0$$이며, 그 역도 성립한다.
- $$D_{KL}(\cdot)$$의 값은 항상 0보다 크다.
- 대칭성이 성립하지 않는다. 즉 $$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$이다. 이러한 특성 때문에 KLD는 두 분포 간의 차이라고는 할 수 있어도, 거리라고는 할 수 없다.

### KLD and Jensen's inequality

쿨백 라이블러 발산은 항상 0보다 크며, 두 분포가 동일할 때에만 그 값이 0이다. 이는 **젠슨 부등식**(Jensen's inequality)을 통해 증명이 가능하다. 젠슨 부등식은 다음과 같다.

$$
if \ f(x) \ is \ convex \quad then \ E[f(x)] \geqq f(E[x])
$$

이를 KLD 수식에 적용하면 다음과 같다.

$$
\eqalign{
KLD(p || q)
&= - \int p(x) \log {q(x) \over p(x)}dx \\
&\geqq - \log \int p(x) \cdot {q(x) \over p(x)} dx = -\log \int q(x) dx = -log 1 = 0
}
$$

이는 $$log$$ 함수가 convex function이기 때문에 가능하다.

### Cross Entropy

위의 쿨백 라이블러 발산 식을 아래와 같이 조금 더 전개해 볼 수 있다.

$$
\eqalign{
D_{KL}(P \| Q) &= \Sigma_x ( P(x)\log{P(x) - P(x) \log Q(x)})\\
&= - \Sigma_x P(x) \log Q(x) - ( - \Sigma_x P(x) \log P(x))\\
&= H(P, Q) -  H(P) \\
}
$$

확률 분포 $$P(x)$$에 따른 확률에 $$Q(x)$$의 정보량 $$- \log Q(x)$$을 곱한 정보량의 총합을 $$H(P,Q)$$로 표기하고 있다. 이를 $$H(P,Q)$$에 대한 식으로 정리하면,

$$
\eqalign{
H(P, Q) &= H(P) + D_{KL}(P \| Q)\\
&= - \Sigma_x P(x) \log Q(x)
}
$$

이 되는데, 이러한 $$H(P,Q)$$를 **cross entropy**라고 한다. 위의 식에서도 알 수 있듯이 cross entropy는 두 분포 간의 KLD 값이 크면 클수록, 즉 두 분포의 차이가 크면 클수록 그 값이 커진다는 특성을 가진다.

#### Cross entropy as Loss Function: Negative Log Likelihood

만약 $$p(x)$$를 고정된 데이터셋 $$(x_1, x_2, ... x_N)$$의 분포라 하고 $$q(x \lvert \theta)$$를 파라미터 $$\theta$$를 갖는 모델이 예측하는 데이터셋의 분포라고 한다면 모델의 학습 방향은 $$\theta$$를 업데이트하여 두 분포 $$p, q$$간의 차이를 줄이는 것이 된다. 이때 두 분포 간의 차이를 다음과 같이 쿨백 라이블러 발산 식으로 표현할 수 있다.

$$
\eqalign{
KLD(p || q_\theta)
&= - \int p(x) \log {q(x \lvert \theta) \over p(x)} dx \\
&\approx - \Sigma_{i=1}^N \log {q(x_i \lvert \theta) \over p(x_i)}\\
&\qquad \qquad (\because E[f] = \int p(x)f(x)dx \approx {1 \over N} \Sigma_{n=1}^N f(x_n))\\
&= \Sigma_{i=1}^N(-\log q(x_i \lvert \theta) + \log p(x_i))
}
$$

이때 $$p(x)$$는 고정되어 있으므로, 두 분포 간의 차이를 줄이기 위해서는 $$- \Sigma_{i=1}^N \log q(x_i \lvert \theta)$$ 를 최소화해야 한다. 이를 **Negative Log Likelihood**라고 한다. 즉, 딥러닝에서 loss를 구하기 위해 자주 사용하는 Negative Log Likelihood를 줄이는 것은 데이터셋의 분포와 현재 모델이 가지고 있는 분포 간의 차이를 줄이는 것을 의미한다.

## Mutual Information

상호 의존 정보란 어떤 사건이 발생했을 때 다른 사건이 발생할 확률에 미치는 영향을 의미한다. 한 마디로 두 확률 변수 간의 독립성을 수치화한 것이라 할 수 있다.

$$
I[X, Y] = KLD(p(X,Y) || p(X), p(Y))
$$

만약 상호 의존 정보 $$I[X, Y] = 0$$이 성립하면, 두 확률 변수 $$X,Y$$는 독립이다.

$$
I[X, Y] = 0 \Leftrightarrow p(X,Y) = p(X), p(Y)
$$

