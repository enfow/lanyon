---
layout: post
title: Pytorch Loss Functions
subtitle: pytorch에서 자주 사용되는 Loss function
---

# Pytorch Loss Functions

- update date: 2020.06.29
- [pytorch documentation](<https://pytorch.org/docs/stable/nn.html>)

## Pytorch Loss function

Pytorch Loss function들은 첫 번째 인자로 prediction value를 받고, 두 번째 인자로 target value를 받는다. 그런데 함수의 유형에 따라 prediction value의 형태와 target value의 형태가 다를 수 있다. 기본적으로 각각의 Loss function은 다음과 같은 특성을 가진다.

|NAME|PRED|TARGET|USAGE|
|:------:|:---:|:---:|:---:|
|MSELoss|(N , * )|(N , * )|Regression|
|NLLLoss|(N , C )|(N)|Multi-Class Classification|
|CrossEntropyLoss|(N , C )|(N)|Multi-Class Classification|
|BCELoss|(N , * )|(N , * )|Binary Classification|

## MSELoss

```python
torch.nn.MSELoss(reduction='mean')
```

딥러닝의 가장 대표적인 손실함수이자 Regression 문제에 많이 사용되는 Mean Square Error를 구해주는 loss function이다. input과 target 두 입력을 받아 둘 간의 MSE loss 를 구해서 반환한다. element 별로 계산이 이루어지므로 input과 target의 형태가 동일해야 한다.

출력의 경우 reduction parameter에 따라 결정되는데 mean이면 말 그대로 전체 loss의 평균으로, sum이면 총합으로 loss가 구해지게 된다. none인 경우에는 input과 동일한 형태로 loss가 개별적으로 구해지게 된다.

```python
pred = torch.Tensor([1,2,3,4,5])
target = torch.Tensor([3,3,3,3,3])

mse = torch.nn.MSELoss(reduction="mean")
mse(pred, target) # tensor(2.)

mse = torch.nn.MSELoss(reduction="sum")
mse(pred, target) # tensor(10.)

mse = torch.nn.MSELoss(reduction="none")
mse(pred, target) # tensor([4., 1., 0., 1., 4.])
```

---

## NLLLoss

```python
torch.nn.NLLLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
    )
```

**Negative Log Likelihood Loss**의 약자로, 말 그대로 Negative Log Likelihood를 구하기 위해 사용하는 loss fucntion이다. Likelihood란 어떤 모수가 주어졌을 때 어떤 값이 그럴 듯한 정도로 정의되는데 여기서는 **각 class 별로 score가 정해졌을 때 해당 score가 그럴 듯한 정도를 의미**한다. 클래스가 1이라면 2번째(0부터 시작하므로) element의 값을 likelihood로 보는 것이다. 이러한 점에서 분류 문제에 자주 사용된다.

```python
pred = torch.Tensor([
    [0.29,0.01,0.7],
])
target = torch.Tensor([2]).long()

nll = torch.nn.NLLLoss()
nll(pred, target)   # tensor(-0.7000)
```

위의 식을 보면 NLLLoss()의 결과가 이름이 의미하는 대로 $$-log(\cdot)$$와는 다르다는 것을 알 수 있다. 정확하게는 음수만을 취하고 있다. [pytorch 공식 문서](<https://pytorch.org/docs/master/generated/torch.nn.NLLLoss.html>)의 NLLLoss()에 관한 설명에서도 아래와 같이 loss가 입력 값의 부호를 바꾼 것으로 정의한다.

$$
l_n = -w_{y_n} x_{n,y_n}
$$

이와 관련하여 [링크](<https://discuss.pytorch.org/t/why-there-is-no-log-operator-in-implementation-of-torch-nn-nllloss/16610>)에서는 $$exp$$를 처리하는 과정에서의 계산상 안정성을 위해 softmax + negative log likelihood 가 아닌 log softmax + negative likelihood 구조를 선택하게 되었고 따라서 NLLLoss()는 뒷부분, 즉 negative likelihood 만을 계산하도록 되었다고 언급하고 있다.

따라서 아래와 같이 pred 값에 log를 취하여 전달해야 Negative Log Likelihood의 값을 구할 수 있게 된다.

```python
pred = torch.Tensor([
    np.log([0.29,0.01,0.7]),
])
target = torch.Tensor([2]).long()

nll = torch.nn.NLLLoss()
nll(pred, target)    # tensor(0.3567)
```

이러한 점에서 Softmax와 Negative Log Likelihood의 조합인 Cross Entropy Loss는 `nn.Softmax`가 아닌 `nn.LogSoftmax`와 `nn.NLLLoss`의 합이 된다.

---

## CrossEntropyLoss

```python
torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
    )
```

MultiClass Classification 문제에 자주 사용되는 **Cross Entropy Loss**는 두 확률 분포의 정보량의 차이를 손실의 크기로 사용하는 것이다.

$$
H_{p,q}(X) = - \Sigma p(x_i) \log q(x_i)
$$

위 식에서 $$p$$의 값은 정답인 경우에만 1이고 나머지는 모두 0이다. 따라서 간단하게 다음과 같이 표현된다.

$$
H_{p,q}(X) = - \log q(x_i)
$$

Cross Entropy를 적용하기 위해 가장 먼저 해야 하는 것은 네트워크의 출력 값을 확률 값에 맞게 0과 1 사이의 값으로 바꾸어주는 것이다. 이를 위해 사용하는 것이 Softmax이다.

$$
Softmax(x_i) = {\exp(x_i) \over \Sigma_j \exp(x_j)}
$$

결과적으로 Cross Entropy는 네트워크 출력 값에 Softmax를 취한 뒤 Negative Log를 씌운 것이 된다. 이를 손실함수로 사용하는 것이 Cross Entropy이다.

LogSoftmax를 취한 뒤 NLLLoss를 한 것과 결과가 동일한 것을 아래와 같이 확인할 수 있다.

```python
# LogSoftmax + NLLLoss
logsoftmax = torch.nn.LogSoftmax()
pred = logsoftmax(torch.Tensor([[0.29,0.01,0.7]]))
target = torch.Tensor([2]).long()

nll = torch.nn.NLLLoss()
nll(pred, target)    # tensor(0.7725)


# CrossEntropyLoss
pred = torch.Tensor([
    [0.29,0.01,0.7],
])
target = torch.Tensor([2]).long()

nll = torch.nn.CrossEntropyLoss()
nll(pred, target)    # tensor(0.7725)
```

---

## BCELoss

```python
torch.nn.BCELoss(
    weight=None,
    size_average=None,
    reduce=None,
    reduction='mean'
    )
```

**Binary Cross Entropy Loss**의 약자로 말 그대로 class가 2개인 상황에서 사용하는 Cross Entropy Loss function이다. loss를 구하는 수식은 다음과 같다.

$$
l_n = -w_n[y_n \log x_n + (1 - y_n) \log(1 - x_n)]
$$

우변에서 weight 값인 $$w_n$$을 제외하고 본다면 베르누이 확률분포에 Negative Log를 취한 것으로 볼 수 있다. Binary Classification의 특성상 어떤 사건이 발생할 확률을 나타내는 $$p$$ 값이 정해지면 다른 사건의 확률도 $$1-p$$로 정해지므로 BCELoss에서는 $$p$$ 값 만을 사용한다. 따라서 BCELoss의 predict는 CrossEntropyLoss와는 달리 0과 1사이의 값이어야 하고, target 값은 0 또는 1의 값을 가지게 된다. 이를 위해 BCELoss를 사용하는 경우 네트워크의 출력에 Sigmoid를 붙여 사용한다.

```python
pred = torch.Tensor([
    [1, 0, 0.5]
])
target = torch.Tensor([1, 0, 1])

nll = torch.nn.BCELoss()
nll(pred, target)    # tensor(0.2310)
```
