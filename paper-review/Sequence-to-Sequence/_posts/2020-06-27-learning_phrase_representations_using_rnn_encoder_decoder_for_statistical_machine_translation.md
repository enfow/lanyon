---
layout: post
title: RNN encoder-decoder) Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
category_num: 1
---

# 논문 제목 : Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

- Kyunghyun Cho, Dzmitry Bahdanau, Fethi Bougares,Holger Schwenk, Yoshua Bengio
- 2014
- [논문 링크](<https://arxiv.org/abs/1406.1078>)
- 2020.06.27 정리

## 세 줄 요약

- Sequence to Sequence model의 기장 기본적인 구조로서 RNN encoder-decoder 구조를 제시한다.
- Encoder에서 입력 sequence 전체의 특성을 추출하면 이를 사용하여 출력 sequence가 결정된다.
- LSTM과 유사하지만 보다 가벼운 GRU 구조를 제시한다.

## Sequence to Sequence

`Sequence to Sequence model`이란 말 그대로 sequence를 받아서 다른 sequence를 출력하는 모델을 말한다. Sequence to Sequence model이 적용되는 대표적인 문제로는 기계번역이 있다. 

<img src="{{site.image_url}}/paper-review/papago.png" style="width:35em; display: block; margin: 0px auto; padding: 15px">

예를 들어 한국어 단어의 sequence라 할 수 있는 다음 문장을

```
이 문장은 Sequence to Sequence 모델을 통해 번역되는 것입니다.
```

영어 문장이자 영어 단어의 sequence인 

```
This sentence is translated through the Sequence to Sequence model.
```

로 바꿔주는 것이다. 논문에서 해결하고자 하는 문제 또한 영어를 불어로 번역하는 것이다.

## RNN Encoder-Decoder architecture

<img src="{{site.image_url}}/paper-review/rnn_encoder_decoder.png" style="width:35em; display: block; margin: 0px auto;">

### RNN(Recurrnet Neural Network)

RNN은 이전 hidden state와 새로운 input 값을 입력으로 받는다.

$$
h_t = f(h_{t-1}, x_t)
$$

위의 수식에서도 확인할 수 있듯 RNN은 $$t$$시점의 입력 $$x_t$$과 함께 순환적(Recurrent)으로 이전 시점 $$t-1$$의 hidden state $$h_{t-1}$$이 $$t$$ 시점의 hidden state $$h_t$$를 결정하는 데에 사용된다. 따라서 **hidden state $$h_t$$에는 과거의 입력 sequence 전체에 대한 정보가 들어있다**고 할 수 있다. 이러한 RNN의 특성에 대해 논문에서는 sequence에 대한 확률분포를 학습할 수 있다라고 표현하고 있다.

- An RNN can learn a probability distribution over a sequence by being trained to predict the next symbol in a sequence.

어떤 문장을 다른 언어로 번역하는 것은 단어와 단어가 순서에 맞춰 1:1로 대응하는 방식으로 이뤄지지 않는다. 즉 언어마다 문법이나 어순이 다르기 때문에 단어와 단어를 서로 매핑하는 식으로는 제대로된 번역이 될 수 없고, 번역하고자 하는 문장 전체를 확인하고 그에 맞춰 새로운 언어의 단어들을 재구성하는 방식으로 이뤄져야 한다. 위의 RNN 설명의 연장선에서 표현하자면 번역하고자 하는 문장의 마지막 hidden state 값이 나온 후에야 번역을 시작할 수 있다는 것이다.

Encoder Decoder 구조는 이러한 특성을 반영하고 있다. 즉 Encoder는 번역하고자 하는 문장 전체에 대한 정보를 담고 있는 hidden state 값을 추출하는 것을 목적으로 하고, Decoder는 Encoder의 hidden state 값을 사용하여 다른 언어의 문장을 구성하는 역할을 수행한다. 이를 위해 RNN Encoder-Decoder 구조에서는 두 개의 RNN을 연결하여 학습하게 된다.

### Encoder

- Variable-length sequence to fixed length vector

Encoder는 입력 sequence의 특성을 반영하는 fixed-length vector를 추출하게 된다. 이를 위해 Encoder는 입력 sequence의 길이에 상관없이 처음부터 끝까지 순차적으로 입력 값을 받게 된다. 그런데 RNN의 hidden state는 최근 입력 값의 영향을 많이 받는다는 특성을 가지고 있으며, 이는 입력 sequence의 길이가 길어질수록 전체 문장의 특성을 최종 hidden state가 제대로 반영할 수 없다는 문제로 이어진다. 이러한 한계를 해결하기 위해 제시된 방법이 Attention mechanism이다.

참고로 마지막 hidden state vector를 부르는 이름은 논문에 따라 다양한데 입력 sequence의 길이에 상관없이 고정적인 길이를 가진다는 점에서 fixed length vector라고 하기도 하고, 전체 문장의 특성을 가지고 있다 하여 context vector라 부르기도 한다. Encoder Decoder 구조에 집중하여 latent라고도 한다.

### Decoder

- fixed length vector to Variable-length sequence

Decoder는 Encoder에서 추출된 context를 통해 새로운 sequence를 만들어내는 역할을 가지고 있다. 만들어낸다고 하지만 Encoder와 크게 다르지 않는 RNN 구조를 가지며 첫 번째 hidden state 입력 값으로 0이 아닌 context vector를 받는다는 점이 가장 큰 차이라고 할 수 있다. 이를 수식으로 표현하면 다음과 같다.

$$
h_t = g(h_{t-1}, y_{t-1}, c)
$$

Encoder와 Decoder는 하나로 묶여 학습이 이뤄진다. 업데이트 식은 다음과 같이 conditional log-likelihood로 표현된다.

$$
\max_{\theta} {1 \over N} \Sigma^{N}_{n=1} \log p_\theta (y_1, ... y_n \lvert x_1, ... x_n)
$$

## New Hideden Unit: GRU

LSTM(Long Short Term Memory)은 Vanilla RNN 구조의 가장 큰 문제로 지적되던 long-term dependency 문제를 long-term state와 short-term state를 구별하는 방식으로 해결했었다. 논문에서는 Hidden Unit that Adaptively Remembers and Forgets라는 이름으로 새로운 RNN model architecture를 제시하고 있는데 이것이 GRU(Gated Recurrent Unit)이다. 

LSTM과 비교해 볼 때 GRU는 비슷한 특성을 유지하면서도 구현이 단순하고 연산량이 적다. 구체적으로 LSTM은 input gate, output gate, forget gate 라는 세 개의 gate를 가지고 있는 반면 GRU는 reset gate와 update gate 두 개 만을 가지고 있다.

GRU의 hidden state 수식은 다음과 같다.

$$
h_t = z h_{t-1} + (1 - z) \tilde h_t
$$

여기서 이전 시점 hidden state $$h_{t-1}$$과 새로운 업데이트 식 $$\tilde h_t$$ 간의 가중치를 결정하는 $$z$$는 update gate에 의해 결정된다. update gate의 수식은 다음과 같다.

$$
z = \sigma([Wx] + [Uh_{t-1}])
$$

그리고 기존 hidden state $$h_{t-1}$$에 더해져서 hidden state의 업데이트 방향이라고 할 수 있는 $$\tilde h$$의 수식은 다음과 같다. 

$$
\tilde h_j = \phi([Wx]_j + [U(r \cdot h_{t-1}])
$$

여기서 r이 reset gate의 출력이며 다음과 같이 계산된다. reset gate 라는 이름은 위의 수식을 보면 쉽게 알 수 있는데, 만약 $$r$$의 값이 0이라면 $$\tilde h$$의 값이 이전 hidden state $$h_{t-1}$$의 영향을 전혀 받지 않게 된다. 이를 통해 이전 데이터를 사용할 것인지 여부 또한 학습할 수 있게 된다.

$$
r = \sigma([Wx] + [Uh_{t-1}])
$$
