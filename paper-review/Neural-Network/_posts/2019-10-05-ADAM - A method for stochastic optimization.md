---
layout: post
title: ADAM) A method for stochastic optimization
---

# 논문 제목 : ADAM) A method for stochastic optimization

- Kingma 등
- 2014
- [논문 링크](<https://arxiv.org/abs/1412.6980>)
- 2019.10.05 정리

## 세 줄 요약

- ADAM은 효과적이고 가장 많이 사용되는 Optimization 방법 중 하나로 Adagrad와 RMSProp 과 관련이 깊다.
- RMSProp과 같이 가중평균을 이용하여 step size를 결정한다. 이를 통해 학습의 정도에 따라 step size를 자동적으로 조절(automatic anealing)할 수 있다.
- 이에 RMSProp과는 달리 가중평균의 결과로 생기는 bias를 제거하는 알고리즘이 포함되어 있다.

## 내용 정리

### ADAM이란

- stochastic optimization 방법 중 하나로 가장 일반적으로 사용되는 방법이기도 하다.
- "Adam is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments"
- first-order optimization method, 즉 1차 미분값만을 사용한다.
  - high-order optimization method는 부적합하다고 지적한다.
- 또한 적은 memory 사용이 장점이다.
- ADAM은 기본적으로 AdaGrad과 RMSProp의 영향을 받았다.
  - AdaGrad는 sparse gradient에서, RMSProp는 on-line, non-stationary setting에서 잘 작동하는 방법이라고 한다.

### AdaGrad

- adagrad의 수식은 다음과 같다.

  $$
  \eqalign{
  &g_t = \nabla(J(\theta))\\

  &G_t = G_{t-1} + (g_t)^2\\

  &\theta_t = \theta_{t-1} - \alpha ( {n \over \root 2 \of {G_t + \epsilon}}) g_t
  }
  $$

- 여기서 $$g_t$$는 objective function의 gradient이다.
- 수식에서 알 수 있듯 누적된 과거 gradient의 변화량($$G_t$$)이 크면 클수록 현재 $$g_t$$ 의 영향력이 작아진다.
- AdaGrad는 iteration이 커지면 step-size가 과도하게 작아진다는 문제점을 가지고 있다.
- 이러한 문제점을 해결하기 위해 제프리 핸튼 등은 **RMSProp**를 제안했다.

### RMSProp

- RMSProp의 수식은 다음과 같다.

  $$
  \eqalign{
  &G_t = \gamma G_{t-1} + (1-\gamma)(g_t)^2\\
  &\theta_t = \theta_{t-1} - \alpha ({ n \over \root \of G_t + \epsilon })g_t
  }
  $$

- AdaGram와 비교해 볼 때 $$G_t$$를 계산하는 부분이 달라졌다. 구체적으로는 단순히 더하던 것이 가중평균(WMA, Weighted Moving Average)으로 바뀌었다.
- 이를 통해 $$G_t$$가 커지는 속도를 조절할 수 있게 되었다.
- 다만 RMSProp는 가중평균을 적용하면서 bias가 생긴다는 문제점을 가지고 있다. 즉, 학습 초기에 WMA는 0부터 시작하게 되므로 실제 gradient의 분포와 학습에 사용되는 gradient의 분포에 차이가 생긴다는 것이다.
  - 이로 인해 인해 수렴이 잘 되지 않는다는 문제가 발생한다고 한다.
- ADAM은 RMSProp의 bias를 제거하는 데에 초점을 맞추고 있다.

### ADAM의 Algorithm

- ADAM의 알고리즘은 다음과 같다.
- 이때 $$\beta_1$$ 로는 0.9, $$\beta_2$$ 로는 0.999 를 일반적으로 사용한다고 한다.
- first moment $$m$$과 second moment $$v$$ 를 사용한다.
$$
\eqalign{
  &m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1)g_t \\
  &v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2)g_t^2
}
$$

- bias를 제거하기 위해 위와 같이 나누기를 실시한다.

$$
\hat m_t = {m_t \over 1 - \beta_1^t } \qquad \hat v_t = {v_t \over 1 - \beta_2^t}
$$

- 결과적으로 RMSProp에서 $$G_t$$ 를 $$\hat v_t$$으로, $$g_t$$ 를 $$m_t$$ 로 바꾼 것과 동일하다.

$$
\theta_t = \theta_{t-1} - \alpha ({\hat m_t \over \root \of {\hat v_t + \epsilon}})
$$

#### ADAM의 upate rule

- $$\epsilon$$이 0 이라고 가정한다면 $$\alpha({m_t \over \root \of {\hat v_t}})$$ 만큼 $$\theta$$가 변화하게 된다(step).
- $$t$$ 시점에서의 step을 $$\Delta t$$ 라고 하자. 그러면 다음과 같은 식이 성립한다.

$$
\eqalign{
&if \ (1 - \beta+1) > \root \of {1 - \beta_2}\\
&\qquad then \ \Delta t \leqq \alpha({1 - \beta_1 \over \root \of {1 - \beta_2}})\\
&elif \ (1 - \beta+1) < \root \of {1 - \beta_2}\\
&\qquad then \ \Delta t \leqq \alpha\\
&elif \ (1 - \beta+1) = \root \of {1 - \beta_2}\\
&\qquad then \ \Delta t < \alpha \quad ... \quad (\because \lvert {\hat m_t \over \root \of {\hat v_t} } \lvert < 1)
}
$$

- 따라서 $$\lvert \Delta t \rvert$$ 의 크기는 $$\alpha$$ 보다 작거나 크게 넘지 않는 수준에서 비슷하다($$\lvert \Delta t \rvert \lessapprox \alpha$$).
- 이를 $$\lvert \Delta t \rvert$$ 의 trust region 이라고 한다.
- 그리고 $$\lvert {\hat m_t \over \root \of {\hat v_t} } \lvert$$ 이 학습의 크기를 결정하는 기준이 되며 이를 **SNR**(signal-to-noise ratio)라고 한다.
  - SNR의 크기가 작으면 작을수록 $$\lvert \Delta t \rvert$$이 0에 가까워진다.
  - optima에 가까울수록 SNR의 크기가 작다. 이러한 점에서 **automatic annealing**이라고 한다.

#### bias correction

- 딥러닝에서 학습을 할 때에는 gradient를 기준으로 weight $$\theta$$ 를 업데이트한다.
- 그런데 RMSProp와 같이 가중평균을 이용하게 되면 gradient를 직접적으로 사용하지 않아 bias가 발생한다. 구체적으로 가중평균의 첫 번째 값이 0이기 때문에 실제 gradient의 분포와 업데이트 되는 값의 분포가 달라진다.
  - ADAM 또한 가중평균을 사용하므로 이러한 문제를 해결해야 한다.
  - ADAM의 알고리즘을 보면 $$g_t$$ 를 $$\hat v_t$$ 으로 대신하는 것을 알 수 있다. 그리고 $$\hat v_t$$ 은 $$v_t$$에서 $$(1-\beta_2^t)$$ 를 나눈 값이다.
  - 즉 $$v_t$$의 분포를 $$g_t$$ 에 맞추기 위해, 그리고 bias를 제거하기 위해 $$1-\beta_2^t$$ 를 나눈 것이라고 할 수 있다. 구체적인 이유는 아래에서 확인할 수 있다.

#####  $$1-\beta_2^t$$ 를 나누어 주는 이유

$$
\eqalign{
&v_t = ( 1 - \beta_2) \Sigma^t_{i=1} \beta_2^{t-i} \cdot g_i^2 \\
& \eqalign{E[v_t] &= [(1 - \beta_2 \Sigma_{i=1}^t) \beta_2^{t-i}\cdot g_i^2]\\
    &= E[g_t^2] (1-\beta_2) \Sigma_{i=1}^t \beta_2^{t-i} + \zeta \\
    &= E[g_t^2] (1-\beta_2) + \zeta
  }
}
$$

- 즉 $$v_t$$ 의 분포와 $$g_t^2$$ 에 $$1 - \beta_2^t$$ 를 곱해준 분포가 비슷하다는 것을 알 수 있다.
