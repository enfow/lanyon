---
layout: post
title: Auto-Encoding Variational Bayes
category_num : 1
keyword: '[VAE]'
---

# 논문 제목 : VAE) Auto-Encoding Variational Bayes

- Diederic Kingma, Max Welling
- 2013
- [논문 링크](<https://arxiv.org/abs/1312.6114>)
- 2020.03.28 정리

## Summary

- VAE는 latent variable $$z$$로 개별 데이터를 생성할 수 있는 Generative Model이다.
- VAE의 **Encoder** 부분은 **regularizer**로, variational inference approximate posterior $$q(z \lvert x)$$를 prior $$p(z)$$에 근사시키는 것과 관련되며, **Decoder** 부분은 latent variable $$z$$를 입력으로 받아 encoder의 입력값으로 주어진 데이터와 비교해 **reconstruction error**를 줄이는 방향으로 업데이트 된다.
- latent variable $$z$$를 직접 샘플링하게 되면 미분이 불가능해 backpropagation이 이뤄질 수 없으며, 이러한 문제를 해결하기 위해 **reparameterization trick**을 사용한다.

<img src="{{site.image_url}}/paper-review/vae.png" style="width: 30em">

## Problem Scenario

**i.i.d condition**을 만족하는 데이터셋 $$X$$가 있고, **latent variable** $$z$$를 이용하여 이를 만들어내는 generative model이 있다면, 아래 수식과 같이 표현할 수 있다.

$$
p_{\theta}(x) = \int p_{\theta}(x \lvert z)p_{\theta}(z) dz
$$

좋은 생성 모델을 만들기 위해서는 $$p_{\theta}(x)$$를 극대화하는 latent variable $$p_{\theta}(z)$$와 parameter $$\theta$$를 찾아야하는데, 논문에서는 이와 관련하여 아레 세 가지 모두 intractable 한 상황을 가정한다. 이로 인해 EM algirithm 과 같은 방법으로는 최적의 경우를 찾을 수 없다.

- marginal likelihood: $$\int p_{\theta}(x \lvert z)p_{\theta}(z) dz$$
- true posterior: $$p_\theta(z \lvert x) = {p_\theta(x \lvert z) p_\theta(z) \over p_\theta(x)}$$
- integral for reasonable mean-field VB algorithm

또한 매우 큰 데이터셋을 가정하기 때문에 Monte Carlo 와 같은 샘플링 기반의 방법론을 곧바로 적용하기에는 시간이 너무 오래 걸리는 상황을 상정한다.

## Variational Inference

marginal likelihood를 곧바로 해결할 수도, $$z$$에 대한 true posterior $$p_\theta(z \lvert x)$$를 구하는 것도 어려운 상황에서 문제를 해결하기 위해 논문의 저자들은 recognition model $$q_\phi(z \lvert x)$$를 도입하여 true posterior $$p_\theta(z \lvert x)$$에 근사시키는 방법을 제시한다. 그리고 이때 사용하는 것이 **VAE**에서 **V**를 의미하는 **Variational Inference** 이다.

Variational Inference는 사후확률분포를 가우시안과 같이 다루기 쉬운 확률 분포로 근사하는 방법이라고 할 수 있다. 베타 분포와 같은 conjugate prior를 가정하지 않는다면 사후확률분포를 구하는 것이 불가능할 정도로 어려운 경우가 많은데, 정확한 사후확률분포를 대신하여 다루기 쉬우면서도(eg. 가우시안) 최대한 실제 사후확률분포와 가까운 것을 찾는 방법 중 하나라고 할 수 있다.

### VI 수식

Variational Inference의 식은 다음과 같다.

$$
\eqalign{ D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x))
&=\int q_{\phi}(z \lvert x) \log {q_{\phi}(z \lvert x) \over p_\theta(z \lvert x)} dz \\
&=\int q_{\phi}(z \lvert x) \log {q_{\phi}(z \lvert x)p_\theta(x) \over p_\theta(x \lvert z) p_\theta(z)} dz \qquad (\because p(z \lvert x) = { p(x \ \lvert z) p(z) \over p(x)})\\
&= \int q_{\phi}(z \lvert x) \log {q_{\phi}(z \lvert x) \over p_\theta(z)}dz + \int q_{\phi}(z \lvert x) \log p_\theta(x) dz - \int q_{\phi}(z \lvert x) \log p_\theta(x \lvert z) dz\\
&= D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z)) + \log p_\theta(x) - E_{z \backsim q_{\phi}(z \lvert x)} [\log p_\theta(x \lvert z)]
}
$$

## Lower Bound

위 Variational Inference 식을 $$\log p_\theta(x)$$에 대해 전개하면,

$$
\log p_\theta(x) = D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x)) - D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(x)) + E_{z \backsim q_{\phi}(z \lvert x)} [log p_\theta(x \lvert z)]
$$

가 된다. 이 식은 다음과 같이 줄여서 표현할 수 있다.

$$
\log p_\theta(x) = D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x)) + L(\theta, \phi ; x)
$$

여기서 $$L(\theta, \phi ; x)$$를 **lower bound**라고 하는데, 이는 우변의 두 번째 항 $$ D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x))$$이 항상 양수이기 때문에 다음과 같이 표현할 수 있어 붙은 이름이다.

$$
\log p_\theta(x) \geqq L(\theta, \phi ; x)
$$

결과적으로 generative model $$p_\theta(x)$$를 구하는 것은 lower bound를 극대화하는 parameter $$\theta, \phi$$를 찾는 문제가 된다.

### Two Ways to express Lower bound

논문에서는 lower bound를 두 가지 방법으로 전개하고 있다.

---

#### Lower bound 1

$$
\eqalign{
    L(\theta, \phi; x)
    &= - D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x)) + \log p_\theta(x)\\
    &= - \int q_{\phi}(z \lvert x) \log {q_{\phi}(z \lvert x) \over p_\theta(x \lvert z)} dx + \int q_{\phi}(z \lvert x) \log p_\theta(z) dz\\
    &= \int q_{\phi}(z \lvert x) (\log p_\theta(z) - \log {q_{\phi}(z \lvert x) \over p_\theta(x \lvert z)}) dz \\
    &= \int \log {p_\theta(z)p_\theta(x \lvert z) \over q_{\phi}(z \lvert x)} \\
    &= \int q_{\phi}(z \lvert x) \log {p_\theta(x,z) \over q_{\phi}(z \lvert x)} \qquad... \quad p_\theta(x,z) = p_\theta(x \lvert z) p_\theta(z)\\
    &= - \int q_{\phi}(z \lvert x) \log q_{\phi}(z \lvert x) dz + \int q_{\phi}(z \lvert x) \log p_\theta(x,z) dz \\
    &= -E_{z \backsim q_{\phi}(z \lvert x)}[\log q_{\phi}(z \lvert x) + \log p_\theta(x,z)]
}
$$

#### Lower bound 2

$$
\eqalign{
    L(\theta, \phi; x)
    &= - D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x)) + \log p_\theta(x)\\
    &= - \int q_{\phi}(z \lvert x) \log {q_{\phi}(z \lvert x) \over p_\theta(x \lvert z)} dx + \int q_{\phi}(z \lvert x) \log p_\theta(z) dz\\
    &= E_{z \backsim q_\phi(z \lvert x)}[\log p_\theta(z) + \log p_\theta(x \lvert z) - \log q_\phi (z \lvert x)]\\
    &= -D_{KL} (q_\phi(z \lvert x) \| p_\theta(z)) + E_{z \backsim q_\phi(z \lvert x)}[\log p_\theta(x \lvert z)]
}
$$

---

두 번째 Lower Bound 식은 직관적이다. 우변의 첫 번째 항 $$-D_{KL} (q_\phi(z \lvert x) \| p_\theta(z))$$은 prior $$p_\theta(z)$$와 이를 모사하기 위해 도입한 $$q_\phi(z \lvert x)$$로 구성되어 있으며, 두 분포의 KL Divergence이기 때문에 근사가 잘 이뤄지면 이뤄질수록 그 크기는 작아지게 되고 그에 맞춰 Lower Bound는 커지게 된다. 이러한 점에서 **Regularization term** 이라고 부른다.

우변의 두 번째 항 $$E_{z \backsim q_\phi(z \lvert x)}[\log p_\theta(x \lvert z)]$$ 의 경우 $$z$$가 $$q_\phi(z \lvert x)$$를 따른다고 할 때 원 데이터에 대한 likelihood 값으로서, **Reconstruction Error Term**이라고 한다. 이는 당연히 크면 클수록 좋다.

## Stochastic Gradient Variational Bayes(SGVB)

Lower bound를 최대로 하는 parameter를 찾기 위해 논문에서는 Stochastic Gradient Variational Bayes를 도입하고 있다. 즉, Auto Encoder와 유사한 구조로 Encoder로 $$q_{\phi}(z \lvert x)$$를, Decoder로 $$log p_\theta(x \lvert z)$$를 모델링하겠다는 것이다. 그런데 여기서 한 가지 문제가 있다면

$$
\text{lower bound} \ L(\theta, \phi ; x) = - D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(x)) + E_{z \backsim q_{\phi}(z \lvert x)} [log p_\theta(x \lvert z)]
$$

에서 우변의 두 번째 항 $$E_{z \backsim q_{\phi}(z \lvert x)} [log p_\theta(x \lvert z)]$$이 $$q_{\phi}(z \lvert x)$$에서 $$z$$를 샘플링하는 함수라는 점이다. 단순히 encoder $$q_{\phi}(z \lvert x)$$에서 샘플링을 해도 되지만, 샘플링에 대해서는 미분을 할 수 없으므로 back propagation이 불가능하다. 즉 encoder의 출력값으로 만들어낸 분포에서 곧바로 샘플링을 진행하여 $$z$$로 사용하면 Gradient Descent를 통한 학습이 이뤄질 수 없다.

### Reparameterization Trick

이러한 문제를 해결하기 위해 도입한 방법이 Reparameterization이다. 쉽게 말해서 $$z \backsim q_\phi(z \lvert x)$$를 변형하여 미분이 가능해지도록 바꾸는 것인데, 논문에서는 다음과 같이 변형하고 있다.

$$
\eqalign{
    & z \backsim q_\phi(z \lvert x) \\
    \rightarrow & \tilde z = g_\phi(\epsilon, x) \qquad \text{with} \ \epsilon \backsim p(\epsilon)
}
$$

이렇게 하면 확률 변수(random variable) $$z$$가 고정적인 값(deterministic value) $$g_\phi(\epsilon, x)$$으로 정의되기 때문에 미분이 가능해진다. 여전히 $$p(\epsilon)$$에서 확률적인 요소가 존재하나, Gradient Descent로 업데이트할 대상은 $$\phi$$이므로 $$p(\epsilon)$$와는 무관하다.

미분이 가능한 형태로 $$z$$를 만들었지만 이것 만으로는 부족하고, 샘플링에 대한 미분이므로 이것을 **Monte Carlo estimate of expectation**에 적용하는 과정이 필요하다. 이를 위해 논문에서는 다음과 같은 과정을 보이고 있다.

---

$$ z = g_\phi (\epsilon, x) $$ 가 주어졌을 때

$$
q_\phi(z \lvert x) \Pi_i dz_i = p(\epsilon) \Pi_i d\epsilon_i
$$

이 성립한다고 할 수 있다. 이 때 $$\Pi_i dz_i$$는 infinitesimal을 고려한 것으로 $$dz$$와 같은 의미라고 생각하면 된다. 이에 따라 아래 수식 또한 성립한다.

$$
\int q_\phi (z \lvert x) f(z) dz = \int p(\epsilon) f(z) d\epsilon = \int p(\epsilon)f(g_\phi(\epsilon, x))d\epsilon
$$

이를 Monte Carlo Estimation에 적용하게 되면 결과적으로 다음과 같다.

$$
\int q_\phi (z \lvert x) f(z) dz \approx {1 \over L} \Sigma^L_{l=1} f(g_\phi(x, \epsilon^{(l)})) \qquad \text{where} \ \epsilon^{(l)} \backsim p(\epsilon)
$$

---

### How to Update

다시 되돌아와 위의 Monte Carlo estimate of expectation 식을 구하게 된 이유는 Lower Bound를 최대가 되도록 업데이트하기 위해서였다. 위에서 Lower Bound를 소개할 때 두 가지로 전개하는 것이 가능하다고 했었는데, 각각에 대해 위 식을 적용할 수 있다. 첫 번째 Lower Bound 전개식에 적용하면 다음과 같다.

$$
L^A(\theta, \phi;X^{(i)}) = {1 \over L} \Sigma^L_{l=1} \log p_\theta(g_\phi(x^{(i)}, z^{(i,l)})) - \log q_\phi (z^{(i, l)} \lvert x^{(i)}) \\
\text{where} \ \epsilon^{(l)} \backsim p(\epsilon) \quad \text{and} \quad z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})
$$

첫 번째 Lower Bound에 대해서 한 것처럼 두 번째 Lower Bound 식에 대해서도 가능하다.

$$
L^B(\theta, \phi;X^{(i)}) = -D_{KL} (q_\phi (z \lvert x^{(i)}) \lvert p_\theta(z)) + {1 \over L} \Sigma^L_{l=1}(\log p_\theta (x^{(i)} \lvert z^{(i,l)})\\
\text{where} \ \epsilon^{(l)} \backsim p(\epsilon) \quad \text{and} \quad z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})
$$

논문에 따르면 두 번째 식을 이용하는 방법이 $$E_{q_\phi(z \lvert x^{(i)})}[\log p_\theta (x^{(i)}\lvert z^{(i,l)})]$$에 대해서만 샘플링을 진행하기 때문에 variance가 더 적다고 한다.

## Connection with Auto-Encoder

$$
L^B(\theta, \phi;X^{(i)}) = -D_{KL} (q_\phi (z \lvert x^{(i)}) \lvert p_\theta(z)) + {1 \over L} \Sigma^L_{l=1}(\log p_\theta (x^{(i)} \lvert z^{(i,l)})
$$

위의 식을 보게 되면 왜 VAE가 Auto Encoder와 같은 형태를 띄게 되었는지를 알 수 있다. 우선 우변의 첫 번째 항은 regularizer로, 두 번째 항은 reconstruction error로서 기능한다는 것은 Lower Bound에서 언급하였다. 이때 $$q_\phi (z \lvert x^{(i)})$$를 Encoder로, $$p_\theta (x^{(i)} \lvert z^{(i,l)})$$을 Decoder로 하면 Auto Encoder의 형태가 된다. latent variable $$z$$의 크기는 input x 보다 작다는 점에서도 그러하다.

그런데 아래의 조건식 때문에 Encoder의 출력값이 바로 $$z$$가 되어 decoder의 입력으로 들어가지는 못한다.

$$\epsilon^{(l)} \backsim p(\epsilon) \quad \text{and} \quad z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})$$

즉, encoder의 출력값에 reparameterization trick을 적용하여 구한 $$z$$를 decoder의 입력으로 넣어주게 된다.

## Variational Auto Encoder

VAE에서는 latent variable $$z$$의 prior $$p_\theta (z)$$를 가우시안 분포로, 그리고 $$p_\theta(x \lvert z)$$를 가우시안 분포 또는 베르누이 분포로 가정할 수 있다. 이렇게 되면 이에 대해 variational inference를 통해 근사하는 approximate posterior $$q_\phi(\cdot)$$ 또한 가우시안 분포로 가정하고 다음과 같이 표현할 수 있다.

$$
\log q_\phi (z \lvert x^{(i)}) = \log N(z; \mu^{(i)}, \sigma^{(i)2} I)
$$

이때 $$\mu^{(i)}, \sigma^{(i)}$$는 endoder 네트워크의 출력값으로 구해진다. 분포의 평균과 표준편차를 알고 있고, 가우시안 분포 임을 가정하고 있기 때문에 다음과 같이 latent variable $$z$$를 구할 수 있다. 참고로 $$\odot$$은 element-wise 곱이다.

$$
z^{(i,l)} = g_\phi(x^{(i)}, \epsilon^{(l)}) = \mu^{(i)} + \sigma^{(i)} \odot \epsilon^{(l)}
$$

그리고 $$q(z), p(z)$$가 모두 가우시안이므로 다음과 같은 공식이 성립한다.

$$
\eqalign{
- D_{KL}( q(z) \| p(z))
&= \int q(z) (\log p(z) - \log q(z)) dz\\
&= {1 \over 2} \Sigma_{j=1}^{J} ( 1 + \log((\sigma_j^{(i)})^2) - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2 )
}
$$

최종적으로 Lower Bound는 다음과 같이 구해진다.

$$
L^B(\theta, \phi;X^{(i)}) \approx {1 \over 2} \Sigma_{j=1}^{J} ( 1 + \log((\sigma_j^{(i)})^2) - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2 ) + {1 \over L} \Sigma^L_{l=1}(\log p_\theta (x^{(i)} \lvert z^{(i,l)}))\\
\text{where} \ z^{(i,l)} = \mu^{(i)} + \sigma^{(i)} \odot \epsilon^{(l)} \quad and \quad \epsilon^{(l)} \backsim N(0, I)
$$
