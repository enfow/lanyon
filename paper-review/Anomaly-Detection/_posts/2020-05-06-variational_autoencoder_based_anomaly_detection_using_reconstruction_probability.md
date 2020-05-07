---
layout: post
title: Variational AutoEncoder based Anomaly Detection using Reconstruction Probability
category_num: 2
---

# 논문 제목 : Variational AutoEncoder based Anomaly Detection using Reconstruction Probability

- Jinwon An, Sungzoon Cho
- 2015
- [paper link](<http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf>)
- 2020.05.06 정리

## Summary

- VAE를 사용하여 구하는 데이터별 Reconstruction Probability를 기준으로 이상 데이터를 탐지하는 알고리즘을 제시한다.
- 하나의 데이터에 있어 복수의 latent를 샘플링하고, 각각에 대한 Decoder의 출력 값을 parameter로 하는 확률 분포를 만든다. 이러한 확률 분포로 reconstruction probability를 구하게 되며, 한 데이터 샘플에 대한 reconstruction probability는 이를 평균한 것으로 한다.
- Encoder와 Decoder 모두 확률 분포를 반환하기 때문에 selective sensitivity 등의 장점을 가지며 이에 따라 여러가지 상황에 유연하게 대처하며 이상 데이터를 탐지하는 것이 가능하다.

## Introduction: Anomaly detection

우리말로 `이상치` 정도로 번역되는 `Anomaly`는 데이터셋을 구성하는 다른 데이터들과 다른 특성을 가지는 데이터라고 할 수 있다. 그리고 `Anomaly detection`은 당연히 이러한 이상치 데이터를 찾아내는 것을 말한다.

Anomaly detection의 방법은 다양한데 논문에서는 크게 statistical based, proximity based, deviation based의 세 가지 방법론으로 나누고 있다.

#### (1) Statistical based Method

Statistical based anomaly detection은 데이터가 정해진 분포에 따라 만들어진다는 가정에서 출발한다. 정상 데이터와 이상 데이터가 서로 다른 분포를 가진다면, 정상 데이터의 확률 분포에서 이상 데이터가 만들어질 확률은 정상 데이터의 그것과 비교해 낮을 수 밖에 없는데 이러한 차이를 이용하는 방법이라고 할 수 있다. 이러한 방법에서는 정상 데이터와 이상 데이터를 구분하는 threshold를 정확하게 설정하는 것이 중요하다.

#### (2) Proximity based Method

Proximity based anomaly detection은 `proximity`라는 표현에서도 알 수 있듯 얼마나 가까운지에 따라 이상 데이터를 찾아내려고 한다. `근접` 여부를 어떻게 측정할 것인가에 따라 다시 세 가지 방법으로 나누어지는데, **(1) Clustering base method**는 데이터를 군집(cluster)별로 분류하고, 군집의 중심(centroid)과의 거리에 따라 이상 데이터를 찾아낸다. **(2) Density based method**는 데이터가 얼마나 몰려 있는가를 기준으로 하며, 데이터가 드문드문 존재하는 영역(sparse region)에 있으면 이상 데이터로 보는 방법이다. 마지막으로 **(3) Distance based method**는 해당 데이터와 가까운 곳에 있는 데이터(neighbor)에 따라 이상치를 판단하는 방법이다. KNN이 대표적이다.

#### (3) Deviation based Method

마지막 Deviation based anomaly detection은 차원 축소와 복원의 과정에서 복원 데이터와 원 데이터 간의 차이, 즉 `reconstruction error`를 기준으로 이상치를 찾아내는 방법이다. 정상 데이터를 잘 복원하는 모델을 만들었다면 정상 데이터에 대해서는 reconstruction errror가 낮게 구해질 것이지만, 정상 데이터와는 다른 분포를 가지는 이상 데이터의 경우 reconstruction error가 높게 나오게 될 것이다. Deviation based Method는 이러한 차이를 기준으로 이상치를 찾는다. 차원 축소에 사용되는 방법으로는 대표적으로 PCA, AutoEncoder가 있다.

## Anomaly Detection with AutoEncoder

`AutoEncoder`는 입력 데이터에 대해 차원 축소를 실시하고 이를 다시 정확하게 복원하는 것을 목표로하는 네트워크 모델이다. Derivation based Method에서는 차원 축소 및 복원을 사용한다고 했었는데, AutoEncoder를 통해 이상치 탐지를 하는 것 또한 가능하다.

### AutoEncoder based anomaly detection

차원 축소를 진행하는 부분을 Encoder라고 하고, 다시 복원하는 부분을 Decoder라고 하며, Encoder의 출력값이자 Decoder의 입력값이 되는 데이터를 Latent 라고 한다. 이때 Encoder, Decoder 수식은 다음과 같다.

$$
\eqalign{
&z = \sigma(W_{en}x + b_{en}) \\
&\hat x = \sigma(W_{de}z + b_{de})
}
$$

학습은 입력 값과 복원 값 간의 차이인 $$\lvert x - \hat x \rvert$$ 를 최소화하는 방향으로 이뤄지는데, 이때 차이를 **reconstruction error**라고 한다. 

AutoEncoder로 이상치 탐지를 실시하는 경우에는 기본적으로 정상 데이터만을 이용하여 학습을 진행해 우선 정상 데이터를 잘 복원하는 모델을 만든 후, 추론 과정에서는 정상 데이터와 이상 데이터를 함께 넣어 reconstruction error가 크게 나오는 것을 이상 데이터로 판단하는 방법을 따른다.

### With Variational AutoEncoder

`Variational AutoEncoder(VAE)`에 관해서는 [링크](<https://enfow.github.io/paper-review/neural-network/2020/03/28/VAE-auto_encoding_variational_bayes/>)에도 정리해 두었다. 사실 VAE의 경우에는 AutoEncoder와 형태만 비슷할 뿐 발전 과정이나 수식의 면에서 큰 차이가 있다. 하지만 AutoEncoder와 마찬가지로 Encoder에 의해 차원 축소가 이뤄지고, 바로 뒤의 Decoder로 이를 복원하며, reconstruction error를 사용하여 학습이 이뤄진다는 점에서 유사하기 때문에 동일한 방법으로 이상치 탐지에 사용할 수 있다.

VAE의 loss function은 다음과 같다.

$$
\eqalign{
    L(\theta, \phi; x)
    &= - D_{KL}(q_{\phi}(z \lvert x) \| p_\theta(z \lvert x)) + \log p_\theta(x)\\
    &= -D_{KL} (q_\phi(z \lvert x) \| p_\theta(z)) + E_{z \backsim q_\phi(z \lvert x)}[\log p_\theta(x \lvert z)]
}
$$

논문에서 제시하고 있는 방법은 AutoEncoder가 아닌 VAE를 사용하고 있다. 이와 관련하여 논문에서는 AutoEncoder의 경우 latent variable이 deterministic 한 반면 VAE는 Probabilistic 한 Encoder를 사용하기 때문에 정상 데이터와 이상 데이터의 평균이 동일하더라도 variance를 통해 구별이 가능하다는 점에서 보다 우수하다고 언급하고 있다.

## Reconstruction probability

논문의 제목에서도 나와있듯 논문에서는 reconstruction error이 아닌 `reconstruction probability`를 사용한다는 것을 강조한다. 즉 입력 데이터와 복원 데이터 간의 차이를 이용하는 것이 아니라, 복원 값의 분포를 따를 때 입력 값이 나올 확률이 얼마나 되는지에 따라서 이상치를 판단하는 것이다. 이를 위해 논문에서는 다음과 같은 새로운 알고리즘을 제시한다.

### Algorithm

1) Encoder에서 평균과 분산을 구한다

$$\mu_z^{(i)}, \sigma_z^{(i)} = f_\theta(z\lvert x^{(i)})$$

2) 정규분포에서 latent variable을 L개 추출한다

 $$\text{draw L samples from} z \backsim N( \mu_z^{(i)}, \sigma_z^{(i)} )$$

3) 각 latent sample을 Decoder의 입력으로 하여 각각에 대한 평균과 분산을 구한다

$$\mu_{(\hat x)}^{(i,l)}, \sigma_{(\hat x)}^{(i,l)} = g_\phi(x\lvert z^{(i,l)})$$

4) 이렇게 얻어진 L개의 분포에 대해 입력 값 $$x^{(i)}$$가 나올 확률을 평균한다

$$\text{recon probability(i)} = {1 \over L} \Sigma_{l=1}^L p_\theta (x^{(i)} \lvert \mu_{(\hat x)}^{(i,l)}, \sigma_{(\hat x)}^{(i,l)} ) $$

5) 확률 값이 $$\alpha$$ 이상인 경우 정상 데이터로, 미만인 경우 이상 데이터로 분류한다.

---

여기서 핵심은 Encoder의 출력값을 parameter로 하여 다수의 latent variable 샘플링하고, 이들을 다시 Deconder에 통과시켜 얻은 출력 값을 parameter로 하는 확률분포로 원본 데이터의 reconstruction probability를 구한다는 것이다. reconstruction error 라면 Decoder의 출력 값을 직접 사용하여 입력 값과의 절대적인 오차를 구했을 것이지만, **논문에서 제시하는 방법은 Decoder의 출력 값으로 확률 분포를 만든다는 점이 가장 큰 차이점**이라고 할 수 있다.

## Difference from an autoencoder based anomaly detection

보다 구체적으로 논문에서는 기존의 AutoEncoder를 사용하는 방법과 논문에서 제시하는 방법 간에는 다음 세 가지 지점에서 차이가 있다고 정리한다.

#### (1) latent variables are stochastic variables

위에서도 잠깐 언급하였지만 AutoEncoder를 사용하는 경우에는 Encoder의 출력 값이 그 자체로 latent variable 이 된다(deterministic). 반면 논문에서 제시하는 방법의 경우 Encoder의 출력 값이 확률 분포의 parameter가 되기 때문에 정확하게 Encoder의 출력 값이 latent variable이 아니라 **latent variable의 확률 분포**라고 할 수 있다(stochastic).

이러한 차이는 정상 데이터와 이상 데이터의 평균이 동일하거나 매우 근접할 때 높은 효과를 보인다. 즉, 정상 데이터와 이상 데이터의 평균이 유사하다면, deterministic method 에서는 latent 또한 같이 유사해진다. 하지만 stochastic method는 mean과 함께 variance 또한 사용하기 때문에 둘 간의 차이를 찾을 수 있게 될 것이다. 논문의 알고리즘에서 latent variable을 복수로 추출하여 모두 사용하는 이유가 variance에서 오는 차이를 잡기 위해서라고 생각할 수 있는 부분이다.

#### (2) reconstructions are stochastic variables

latent variable 뿐만 아니라 reconstruction, 즉 Decoder의 출력 값 또한 확률 분포라는 점에서도 기존의 방법과 다르다. 논문에서는 이것의 장점을 `selective sensitivity`라고 표현한다. 이것이 가능한 이유 또한 variance 덕분인데 variance가 높은 경우에는 mean과의 차이가 어느 정도 있더라도 reconstruction probability 가 높게 나올 수도 있는 반면, variance가 낮은 경우에는 mean과의 차이가 작아도 reconstruction probability 가 크게 낮아질 수 있기 때문이다. 이렇게 되면 데이터에 따라 정상 이상을 구별하는데 있어 중요한 것과 그렇지 않은 것을 학습할 수 있고, 이에 대한 정보를 추론에도 사용할 수 있다.

#### (3) reconstructions are probability measures

Reconstruction error의 크기를 기준으로 하는 경우에는 객관적인 threshold(알고리즘 상 $$\alpha$$)를 어떻게 잡을 것인가가 문제 된다. 특히 이러한 문제는 여러 종류의 정상 데이터가 혼재하고 있는 경우(heterogeneous)에 크게 나타난다. 여러 종류의 정상 데이터가 있다면 각각의 데이터 종류에 맞는 threshold를 가중평균하여 기준 threshold를 설정하는 방법이 고려될 수 있지만, 이것은 객관적인 방법도 아니고 모든 데이터에 대해 적절한 방법도 아니다.

하지만 Reconstruction probability를 사용하면 이러한 문제가 줄어든다. 왜냐하면 여러 종류의 데이터가 있다 하더라도 해당 데이터가 reconstruction 될 확률은 모두 동일한 비중으로 평가될 수 있기 때문이다. 이를 두고 논문에서는 **"1 % probability is always a 1 % for any data"** 라고 표현한다. 이러한 상황에서는 보다 객관적이고 합리적인 threshold의 설정이 가능하다.
