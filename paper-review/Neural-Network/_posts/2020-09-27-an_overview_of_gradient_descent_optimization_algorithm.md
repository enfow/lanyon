---
layout: post
title: An overview of gradient descent optimization algorithms
category_num : 1
keyword: '[Optimizer]'
---

# 논문 제목 : An overview of gradient descent optimization algorithms

- Sebastian Ruder
- 2017
- [논문 링크](<https://arxiv.org/abs/1609.04747>)
- 2020.09.27 정리

## Summary

- Mini-Batch의 사이즈가 크면 클수록 시간과 메모리를 많이 소요하나, Gradient의 Variance가 적어 안정적인 업데이트가 가능하다. 두 가지의 Trade-off를 고려하여 Batch Size를 결정하는 것이 좋다.
- Non-Convex Surface에서 Gradient Descent Method로 업데이트 할 때 Local Minima 보다는 Saddle Point로 인한 어려움이 더 크다.
- Adaptive Learning Rate Method로는 Adam, RMSprop, AdaDelta, AdaGrad 등이 있으며, 이 중 Bias-Correction이 추가되어 있는 Adam이 가장 좋은 경우가 많다.

## Too Many Options

딥러닝의 문제점 중 하나는 선택해야 하는 것이 너무 많다는 것인데, Mini Batch의 크기는 어떻게 할 것인지와 어떤 Optimizer를 쓸 것인지도 이에 포함된다. Mini Batch의 크기를 극단적으로 너무 크거나 작게 가져가는 경우는 많지 않지만, Task의 종류에 따라 Batch Size에 민감하게 반응하기도 한다. Optimizer는 일반적으로 Adam을 많이 사용하지만 경우에 따라서는 Adagrad, RMSprop 또는 Vanilla SGD가 더 좋은 결과를 낳기도 한다. 이러한 점에서 Mini Batch의 크기는 어떻게 할지, 그리고 Optimzer는 무엇으로 할지 결정하는 것은 딥러닝에서 중요한 문제 중 하나다.

## Gradient descent variants

Mini Batch의 크기를 결정하는 것은 결국 한 번 Gradient를 계산하는 데 있어 몇 개의 Data Point를 볼 것인가에 관한 문제라고 할 수 있다. 일반적으로 데이터 포인트의 갯수에 따라 업데이트의 정확도와 시간과 메모리와 같이 소요되는 자원의 크기는 서로 Trade-Off 관계에 있다. 즉 한 번에 많은 데이터를 본다면 Gradient의 Variance가 낮아 업데이트는 안정적이지만 시간이 오래 걸리고 메모리를 많이 차지하지만, 적은 데이터를 본다면 반대로 Gradient의 Variance는 커지되 빠르고 적은 메모리로도 업데이트가 가능하다.

Mini Batch의 극단적인 케이스로 한 번에 데이터 셋에 포함된 모든 데이터 포인트를 사용하는 것과, 한 번에 단 하나의 데이터 포인트만 사용하는 것을 생각해 볼 수 있다. 이때 한 번에 모든 데이터 포인트를 사용하는 방법을 Batch Gradient라고 하고, 한 번에 하나의 데이터 포인트를 사용하는 방법을 Stochastic Gradient Descent라고 한다.

### Batch Gradient Descent

Vanilla Gradient Descent라고도 하는 **Batch Gradient Descent**는 Gradient Descent를 구현하는 가장 기본적인 방법이다. 한 번의 업데이트를 위해 데이터셋의 모든 데이터 포인트를 사용하기 때문에 Gradient 계산에 시간이 오래 걸리고 메모리가 많이 필요하다. 그리고 데이터가 실시간으로 들어오는 경우, 즉 online setting에서는 전체 데이터셋을 확정할 수 없으므로 사용할 수 없다.

$$
\theta = \theta - \eta \cdot \Delta_\theta J (\theta)
$$

Batch Gradient Descent는 Convex Surface에서는 Global Minima로의 수렴이 보장된다. Non-Convex Surface에서는 Local Minima로의 수렴이 보장된다.

### Stochastic Gradient Descent

Stochastic Gradient Descent는 Batch Gradient Descent와 정반대로 데이터 포인트 하나하나를 대상으로 Gradient를 구하고 업데이트하는 방법을 말한다. 따라서 Batch Gradient Descent와 장단점이 완전히 바뀌어, 업데이트의 속도가 빠르고 메모리 소모량은 적다는 장점을 가진다. 하지만 개별 데이터 포인트를 대상으로 하기 때문에 업데이트의 Variance가 크고, 이로 인해 업데이트 과정에 **Fluctuation**이 심하다는 단점이 있다.

$$
\theta = \theta - \eta \cdot \Delta_\theta J(\theta; x^{(i)}, y^{(i)})
$$

업데이트의 Variance가 크다는 것은 양날의 검이라고 할 수 있다. Gradient의 방향이 빈번하게 바뀌기 때문에 더 좋은 Local Minima를 찾을 가능성이 있는 반면, **Overshooting**으로 인해 정확한 Local Minima에 수렴하는 것 자체가 어렵다는 문제가 자주 발생한다. 이를 극복하기 위해 업데이트가 진행됨에 따라 점차 learning rate를 줄여나가는 방법을 사용하기도 한다.

### Mini Batch Gradient Descent

앞서 언급한 두 방법의 타협점이자, 현재 딥러닝에서 가장 일반적으로 사용되는 방법이다(이러한 점에서 SGD라고 하면 Mini Batch Gradient Descent를 의미하는 경우가 대부분이다). Variance를 줄여 안정적인 학습을 가능하게 하면서도, 업데이트의 속도를 빠르게 할 수 있다.

$$
\theta = \theta - \eta \cdot \Delta_\theta J(\theta; x^{(i,i+n)}, y^{(i,i+n)})
$$

일반적으로 Batch Size는 50에서 256 사이로 결정하나, 경우에 따라서는 다양한 크기가 적용된다고 한다.

#### Challenges of Mini Batch Gradient Descent

하지만 Mini Batch Gradient Descent를 그대로 사용하면 학습이 잘 되지 않는다. 이와 관련하여 다양한 원인이 지목되는데, 첫 번째는 learning rate이다. 고정적인 learning rate를 사용하는 경우 너무 작게 잡으면 수렴 속도가 느리고, 너무 크게 잡으면 Local Minima 주변에서 횡보하거나, 발산해버리는데, 문제는 각 Task마다 적절한 learning rate를 알기란 매우 어렵다는 것이다. 앞서 언급한대로 학습이 진행됨에 따라 learning rate를 줄이는 방법도 있으나, 이 또한 언제 얼마나 줄일 것인가를 결정해야 한다는 문제가 있다.

두 번째는 Non-Convex와 관련된 문제이다. Non-Convex라 한다면 Local Minima가 다수 존재하여 그 중 하나에 빠져버리는 문제만을 생각하기 쉬운데, 사실 학습과 관련된 많은 문제들이 Saddle Point로 인해 발생한다고 한다. Saddle Point 주변에서는 Gradient가 거의 0이 되기 때문에 여기에서 빠져나오는 것이 쉽지 않다는 것이 안정적인 학습을 방해한다는 것이다.

## Gradient Descent Optimization Algorithms

이러한 문제들을 극복하기 위해 단순히 업데이트에 사용하는 데이터 포인트의 갯수 외에도 다양한 방식으로 Gradient의 크기를 조절하는 알고리즘들이 제시되었다. 논문에서는 다음과 같이 Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam 등에 대해 언급하고 있다.

### Momentum

**Momentum**은 Vanilla SGD를 사용하면 골짜기를 횡보하며 내려가는 문제를 직관적으로 해결하려 한다. 골짜기에서 횡보하는 것은 Gradient의 방향과 Local Mimima로의 방향이 불일치하기 때문에 발생한다. 스키를 타고 산을 내려오는 것에 비유하자면, 가장 빠르게 내려가는 방법은 가장 낮은 곳을 향해 직선으로 나아가는 것이 될 것이다. 하지만 Gradient Descent는 Local Minima를 향해 가는 것이 아니라 각 지점에서 가장 크게 내려갈 수 있는 방향으로 가는 방법이다. 이 경우 장기적으로 본다면 Local Minima 방향으로 간다고 할지라도 업데이트의 경로가 최단 경로라고는 할 수 없다.

이 과정에서 Momentum은 장기적인 추세는 Local Mimima 방향에 가까울 것이라는 점에 집중한다. 즉 현재의 Gradient 뿐만 아니라 지금까지 업데이트되고 있는 방향을 감안해 전체적인 업데이트 방향을 결정하여 조금 더 Local Minima에 빠르게 다가가도록 하겠다는 것이다.

$$
\eqalign{
v_t &= \gamma v_{t-1} + \eta \Delta_\theta J(\theta) \\
\theta &= \theta - v_t
}
$$

Momentum Term $$\gamma v_{t-1}$$의 비율을 결정하는 $$\gamma$$는 0.9 또는 그와 비슷한 값으로 설정하는 것이 일반적이다. 재귀적이므로 $$v_t$$의 값은 $$v_1$$ 시점부터 현재 $$v_t$$시점까지의 Momentum이 모두 반영된다.

### NAG

Momentum의 문제점 중 하나는 Local Minima에 도달했을 때에도 기존의 Momentum의 영향으로 인해 이에서 이탈하여 수렴하는 것이 어려워진다는 점이다. 스키를 타고 산비탈을 모두 내려왔지만 관성으로 인해 반대편 산비탈을 올라가는 상황을 생각해 볼 수 있다. **NAG(Nesterov Accelerated Gradient)**는 이러한 방법을 Gradient를 계산하는 위치를 바꾸어 해결하려고 한다.

$$
\eqalign{
&v_t &= \gamma v_{t-1} + \eta \Delta_\theta J(\theta - \gamma v_{t-1}) \\
&\theta &= \theta - v_t
}
$$

위의 수식을 Momentum의 수식과 비교해보면 $$\eta \Delta_\theta J(\theta)$$가 $$\eta \Delta_\theta J(\theta - \gamma v_{t-1})$$로 바뀌었음을 알 수 있다. 즉 Gradient를 구할 때 현재의 위치($$\theta$$)가 아닌, Momentum Term의 크기만큼 이동한 후의 위치($$\theta - \gamma v_{t-1}$$)를 기준으로 하겠다는 것이다.

### AdaGrad

**AdaGrad**와 Momentum, NAG 간의 가장 큰 차이점은 $$\theta$$를 구성하는 개별 파라미터마다 서로 다른 Learning Rate가 적용된다는 것이다. 구체적으로 지금까지 많이 업데이트 된 파라미터에 대해서는 작은 Learning Rate를, 적게 업데이트 된 파라미터에 대해서는 큰 Learning Rate를 적용하게 되는데, 이러한 점에서 **Ada**ptive **Grad**ient라고 한다. 이를 수식으로 살펴보면 다음과 같다. 참고로 $$\epsilon$$은 Smoothing term으로, 분모가 0이 되는 것을 방지하기 위해 더해주는 매우 작은 수($$1e-8$$)이다.

$$
\eqalign{
g_{t, i} &= \Delta_{\theta_t} J(\theta_{t, i})\\
\theta_{t+1, i} &= \theta_{t, i} - {\eta \over {\root \of {G_{t, ii} + \epsilon}}}\cdot g_{t, i}
}
$$

$$\theta_{i, t}$$는 $$t$$시점의 $$i$$번째 파라미터를, $$G_t \in \mathcal R^{d \times d}$$는 $$t$$시점까지 $$\theta_i$$에 적용된 Gradient의 제곱합을 대각요소로 가지는 대각행렬을 의미한다. 즉 $$G_{t, ii}$$는 $$\theta_i$$의 누적 Gradient 제곱합이라고 할 수 있다. 이러한 점을 이용하여 다음과 같이 Element-Wise Product($$
\odot$$)를 사용해 하나의 식으로도 표현할 수 있다.

$$
\theta_{t+1} = \theta - {\eta \over {\root \of {G_t + \epsilon}}} \odot g_t
$$

이로인해 Learning Rate를 업데이트가 진행됨에 따라 임의로 조정해줄 필요가 없어졌다. 그리고 파라미터마다 다른 Learning Rate가 적용되고, 특히 학습이 덜 이뤄진 파라미터에 대해서 크게 업데이트 되도록 하고 있어 AdaGrad의 성능은 sparse data를 학습할 때 더욱 도드라진다고 한다. 다만 AdaGrad는 분모가 Gradient의 제곱의 누적합이기 때문에 계속해서 커지고, 이로 인해 Learning Rate가 계속해서 작아진다는 문제를 가지고 있다.

### AdaDelta & RMSprop

AdaGrad에서 Learning Rate가 계속 작아지는 문제의 원인은 과거의 모든 Gradient가 동일한 가중치를 가지고 있다는 것에 있다. **AdaDelta**는 Gradient의 누적 제곱합이 아닌 다음과 같이 가중 평균을 적용하여 재귀적으로 과거의 Gradient가 미치는 영향을 줄여 이러한 문제를 해결하려 한다. 이때 $$\gamma$$는 0.9로 설정하는 것이 일반적이라고 한다.

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2
$$

업데이트 식은 $$G_t$$가 $$E[g^2]_t$$로 대체되었다는 것 외에는 차이가 없다.

$$
\theta_{t+1} = \theta_t - {\eta \over {\root \of {E[g^2]_t + \epsilon}}} \odot g_t
$$

**RMSprop**는 Goeff Hinton이 강의에서 제시한 방법으로 여기까지의 Adadelta와 동일하다. 그런데 AdaDelta는 여기서 멈추지 않고 조금 더 나아가, 다음과 같이 Gradient의 제곱을 사용하는 것이 아니라 파라미터의 변화율의 제곱의 평균을 사용하는 방법을 제시한다.

$$
E[\Delta \theta^2]_t = \gamma E[\Delta \theta^2]_{t-1} + (1 - \gamma) \Delta \theta_t^2
$$

이를 통해 아래 수식과 같이 Default Learning Rate $$\eta$$ 없이도 업데이트가 가능하다고 한다.

$$
\eqalign{
\Delta \theta_t &= - {\root \of {E[\Delta \theta^2]_{t-1} + \epsilon} \over \root \of {E[g^2]_{t} + \epsilon}}\\
\theta_{t+1} &= \theta_t + \Delta \theta_t
}
$$

### Adam

**Adaptive Moment Estimation**의 약자인 **Adam**은 가장 일반적으로 사용되는 Optimizer라고 할 수 있다. Adam 또한 AdaGrad, AdaDelta 등과 같이 각 파라미터에 적용되는 Learning Rate를 다르게 하여 업데이트하는데, 이를 결정하는 방식에서 차이가 있다. 구체적으로 Gradient의 제곱 가중 합($$v_t$$, Second Moment) 뿐만 아니라 Gradient의 제곱 가중 합($$m_t$$, First Moment)을 함께 사용한다.

$$
\eqalign{
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
}
$$

업데이트 식은 AdaGrad와 크게 다르지 않고, 크게 보면 Gradient의 자리에 First Moment $$m_t$$가 대신한다는 점에서만 차이가 있다.

$$
\theta_{t+1} = \theta_t - {\eta \over {\root \of v_t} + \epsilon} \cdot m_t
$$

하지만 Adam의 저자들은 이와같이 곧바로 적용하는 경우 $$m_t$$와 $$v_t$$의 초기값에 의한 Bias가 크게 나타난다는 문제을 확인했다고 한다. 즉 $$m_t = 0, v_t = 0$$으로 했을 때 $$0$$에 대해 Bias가 발생하여 정확한 업데이트 방향을 결정하는 데에 어려움을 겪는다는 것이다. 이와 같은 문제를 해결하기 위해 다음과 같이 First Moment와 Second Moment의 값을 조정하게 된다.

$$
\eqalign{
\hat m_t &= {m_t \over 1 - \beta_1^t}\\
\hat v_t &= {v_t \over 1 - \beta_2^t}\\
}
$$

최종 업데이트 식은 다음과 같다.

$$
\theta_{t+1} = \theta_t - {\eta \over {\root \of {\hat v_t}} + \epsilon} \cdot \hat m_t
$$

참고로 $$\beta_1$$로는 0.9를, $$\beta_2$$로는 0.999를 사용하는 것이 일반적이다.

### Which Optimizer to Use

결론적으로 이야기하면 Adam을 사용하는 것이 가장 좋다. RMSprop, AdaDelta, Adam은 서로 매우 유사한 알고리즘이고, 따라서 비슷한 환경에서 유사한 성능을 보이지만 Kingma 등에 따르면 Adam의 Bias-Correction이 Gradient가 Sparse해지는 경우에 효과적이므로, Adam이 가장 좋다는 것이다. Adaptive Learning-Rate Method를 사용하지 않고, 순수한 SGD를 통해 업데이트 하는 경우도 많지만 수렴에 시간이 오래 걸리고 Saddle Point에 취약하므로 좋은 방법은 아니라고 논문에서는 결론짓고 있다.
