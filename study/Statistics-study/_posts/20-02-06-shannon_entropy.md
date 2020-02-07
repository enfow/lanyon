---
layout: post
title: Shannon Entropy
category_num: 3
---

# Shannon Entropy

- Ian Goodfellow, Yoshua Bengio, Aaron Courville의 Deep Learning Book을 참고하여 작성했습니다.
- update date : 2020.02.06

## Information Theory

정보이론은 어떤 signal이 주어졌을 때 그에 담긴 정보의 양이 얼마나 되는지에 대해 관심을 갖는 응용 수학 분야이다. 정보이론은 자주 발생하지 않는 특수한 사건이 벌어졌다는 것을 알았을 때 빈번하게 발생하는 사건보다 더 많이 놀라게 된다는 기본적인 아이디어에서 출발한다. 보다 구체적으로 정보이론은 다음 세 가지 가정을 바탕으로 한다.

1) 발생 빈도가 높은 사건은 적은 정보를 가지고 있다.

2) 발생 빈도가 낮은 사건은 많은 정보를 가지고 있다.

3) 독립 사건은 추가적인 정보가 된다.

### self-information

위의 가정에 맞게 어떤 단일 사건 $$X = x$$에 대해 그것이 가지는 정보량을 아래와 같이 정의할 수 있다.

$$
I(x) = - \log P(x)
$$

이때 로그의 밑을 자연상수 $$e$$ 또는 2로 한다.

이와 같이 확률 값에 음의 로그를 취하게 되면 확률이 높으면 높을수록 낮은 값을, 낮으면 낮을수록 높은 값을 반환하게 되므로 위의 가정에 부합한다고 할 수 있다.

### shannon entropy

위에서 정의한 self-information은 말 그대로 어떤 사건이 발생했다는 정보가 가지는 정보량이다. 단일 사건이 아닌 어떤 분포의 정보량 또한 아래와 같이 구할 수 있다.

$$
\eqalign{
H(X) &= E_{X \backsim P}[I(x)]\\ &= - E_{X \backsim P}[ \log P(x) ]\\ &= - \Sigma_x P(x) \log P(x)
}
$$

위의 식은 확률 분포 $$P$$의 정보량이라는 뜻에서 $$H(P)$$로도 표기하기도 하며, 어떤 분포에 따라 사건이 발생했다는 것을 알았을 때 기대되는 정보량으로 정의된다. 이와 같이 어떤 확률 분포가 가지는 정보량을 **Shannon Entropy**라고 한다.

### Kullback Leibler Divergence

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
- $$D_{KL}(\cdot)$$의 값은 항상 0 이상이다.
- 대칭성이 성립하지 않는다. 즉 $$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$이다. 이러한 특성 때문에 KLD는 두 분포 간의 차이이지, 거리라고 할 수 없다.

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

#### 손실함수로서 Cross entropy

cross entropy는 두 분포 $$P(X)$$와 $$Q(X)$$간의 차이를 줄이고 싶을 때 손실함수로 많이 사용된다. 만약 $$P(X)$$를 고정된 전체 데이터셋의 분포라 하고 $$Q(X)$$를 모델이 예측하는 데이터셋의 분포라고 한다면 모델의 학습 방향은 두 분포 간의 차이를 줄이는 것이 된다. 이때 두 함수 간의 차이를 cross entropy 함수로 구할 수 있으며, 이를 손실함수로 사용하여 두 분포 간의 차이를 줄이는 방향으로 학습이 가능하다.
