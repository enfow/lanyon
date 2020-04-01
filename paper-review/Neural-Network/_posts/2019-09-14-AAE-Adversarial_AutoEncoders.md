---
layout: post
title: AAE) Adversarial AutoEncoders
category_num : 4
---

# 논문 제목 : AAE) Adversarial AutoEncoders

- Alireza Makhzani 등
- 2016
- [논문 링크](<https://arxiv.org/abs/1511.05644>)
- 2019.09.14 정리

## Summary

- AAE(adversarial autoencoder)는 AE(autoencoder)와 GAN(generative adversarial networks)를 결합한 모델이라고 할 수 있다.
- VAE에서는 KL term을 계산하기 위해 prior의 density model을 gaussian 등으로 미리 정의해야하지만 AAE에서는 이를 GAN loss로 대체한다.
- 이러한 점 때문에 VAE와는 달리 보다 다양한 prior를 가정할 수 있다.

## Adversarial AutoEncoder

- Auto Encoder 구조를 갖는 Generative model이다.
- AAE에서 AE는 두 개를 기준으로 학습한다.
    1. traditional reconstruction error criterion
    2. adversarial training criterion
        - "matches the aggregated posterior distribution of the latent representation of the AE to an arbitrary prior distribution"

- "The AAE is an AE that is regularized by matching the aggregated posterior $$g(z)$$, to an arbitrary prior $$p(z)$$"
  - "guides $$q(z)$$ to match $$p(z)$$"
- "The encoder ensures the aggregated posterior distribution can fool the discriminative adversarial network into thinking that the hidden code $$q(z)$$ comes from the true prior distribution $$p(z)$$"
  - 한 마디로 말해 $$q(z)$$에서 추출한 sample을 $$p(z)$$에서 추출한 것으로 보이게 한다.

## AAE의 Training

- AAE의 training은 크게 두 개의 phase로 나뉜다.
    1. Reconstruction phase(AE)
        - AE에서 하는 것처럼 encoder와 decoder를 학습한다.
    2. Regularization phase(Adversarial networks)
        - 다시 두 단계로 나누어진다.
            1. $$p(z)$$와 $$q(z)$$를 구분할 수 있도록 discriminator를 학습한다.
            2. $$p(z)$$와 $$q(z)$$를 구분하기 어렵도록 generator를 학습한다.

## Encoder $$q(z \lvert x)$$의 세 가지 종류

1. Deterministic
    - pd(x)를 통해서만 $$q(z)$$를 유추한다.
    - AE와 유사하다.
2. Gaussian posterior
    - $$q(z \lvert x)$$가 가우시안을 따른다고 가정한다.
    - stochasticity in $$q(z)$$가 data distribution 과 randomness of gaussian distribution at the output of encoder로 결정된다.
    - VAE와 마찬가지로 reparameterization trick을 사용한다.
3. Universal approximator posterior
    - encoder가 x와 노이즈 n를 input으로 받는 fixed distribution으로 가정한다.
    - 추가적으로 공부 필요

## AAE와 VAE의 비교

- VAE는 prior를 알아내기 위해서 KLD를 사용한다.
- AAE는 aggregated posterior of the hidden code vector를 prior posterior에 맞추도록 하여 비슷한 효과를 낸다.
- 즉 KLD 대신 GAN을 도입했다고 할 수 있다.
- VAE와 AAE의 가장 큰 차이점은 prior distribution의 표현에 있다. 즉, VAE는 monte carlo에 의한 KLD 계산이 필요하므로 prior distribution을 정확히 어떤 함수로 정의(exact function)해야 한다 - (Explicit density). 반면 AAE는 prior를 정확히 정의하지 않고 단순히 sampling만 가능하면 된다(Implict density, GAN의 특성).
- VAE의 regularization term을 GAN loss로 대체할 경우 posterior와 prior 가 정규분포 외에 다른 확률분포를 사용할 수 있게 되기 때문에 모델 선택의 폭이 넓어진다.
