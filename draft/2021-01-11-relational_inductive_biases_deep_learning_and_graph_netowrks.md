---
layout: post
title: Relational Inductive Biases, Deep Learning, and Graph Networks
category_num : 2
---

# 논문 제목 : Relational Inductive Biases, Deep Learning, and Graph Networks

- Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst 등
- 2018
- [Paper Link](<https://arxiv.org/pdf/1806.01261.pdf>)
- 2021.01.11 정리

## Summary

## Inductive Bias

딥러닝 모델에는 다양한 기본 구조들이 존재한다. 가장 단순하면서도 기본적인 구조라고 할 수 있는 **Fully Connected Network(FCN)**, 이미지를 다루는 분야에서 많이 사용되는 **Convolution Neural Network(CNN)**, 언어를 비롯한 시계열 데이터에서 효과적인 **Recurrent Neural Network(RNN)** 등이 가장 널리 알려진 구조들이다. 특정 분야에서 가장 효과적이라고 알려진 딥러닝 모델들은 결국 이러한 기본 구조들을 적절하게 조합한 결과라고 할 수 있다.

사실 위에서 언급한 각각의 구조들이 가지는 상대적인 장점들은 잘 알려져 있다. 이미지가 대표적이라고 할 수 있을텐데, 기본적인 MNIST Classification에서 시작하여 Semantic Segmentation, Facial Recognition까지 이미지를 다루는 대부분의 딥러닝 모델들은 모두 CNN을 사용한다.

**Inductive Bias**는 이와 같이 CNN이 왜 이미지를 다루는 작업에 있어 강점을 보이는지에 대한 설명이라고 할 수 있다. 여기서 말하는 Inductive Bias는 다음과 같이 정의된다([Springer](<https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_927>)).

- In machine learning, the term inductive bias refers to a set of (explicit or implicit) assumptions made by a learning algorithm in order to perform induction, that is, to generalize a finite set of observation (training data) into a general model of the domain.

쉽게 말해 Training에서 보지 못한 데이터에 대해서도 적절한 귀납적 추론이 가능하도록 하기 위해 알고리즘(모델)이 가지고 있는 가정들의 집합이라는 것이다. 예를 들어 선형 회귀 모델을 사용한다는 것은 입력 데이터에 선형성이 존재한다는 것을 확인했거나, 혹은 선형성을 띈다고 가정해도 충분하다고 생각하기 때문이라고 할 수 있을 것인데, 이러한 것들이 Inductive Bias의 대표적인 예시가 된다.

### Relational Inductive Bias

Inductive Bias는 크게 **Relational Inductive Bias**와 **Non-relational Inductive Bias** 두 개로 나뉜다. 이때 Relational Inductive Bias는 말 그대로 Inductive Bias 중에서도 어떤 관계에 초점을 맞춘 것이라고 할 수 있는데, 여기서 말하는 관계란 입력 Element와 출력 Element 간의 관계를 말한다.

<img src="{{site.image_url}}/paper-review/relational_inductive_bias_table.png" style="width:48em; display: block; margin: 2em auto;">

<img src="{{site.image_url}}/paper-review/relational_inductive_bias_compare_buliding_blocks.png" style="width:48em; display: block; margin: 2em auto;">

위의 표와 그림을 보면 보다 명확한데, FCN은 입력 Entities가 개별 Unit으로 정의되며, 이들은 서로 모두 연결되어 있는 것으로 가정한다(All-to-all Relations). 모든 입력 Element가 모든 출력 Element에 영향을 미친다는 점에서 구조적으로 특별한 Relational Inductive Bias를 가정하지 않는다(Weak).

반면 CNN은 입력이 이미지처럼 격자(Grid) 구조로 되어 있다. 일반적으로 입력의 크기보다 작은 Convolution Filter를 사용하여 전체의 일부만을 대상으로 Convolution Operation을 수행하여 출력을 결정한다. 이러한 점에서 CNN에서는 Entities 간의 Relation이 지역성, 즉 서로 가까운 Element 간에만 존재한다고 가정하는 것으로 볼 수 있으며, 결과적으로 어떤 특성을 가지는 Element들이 서로 뭉쳐있는지 중요한 경우에 탁월한 구조가 된다(Relational Inductive Bias - Locality). CNN이 Spatial Translation에 강력한(Robust) 이유이기도 하다(Spatial Invariance).



- 보지 못한 데이터에 대해서도 귀납적 추론을 하기 위해 모델(알고리즘)에서 가지고 있는 가정들의 집합. 선형회귀의 선형성 가정, 베이지안 모델에서 주어지는 Prior Distribution 등이 대표적이다.
- Regularization Term 또한 Inductive Bias 중 하나라고 할 수 있다.




딥러닝 구조를 사용하면서 어떻게 Relational Inductive Biases를 잘 사용할 것인지를 탐색한다.

Graph Network는 Relational Reasoning과 Combinatorial Generalization을 제공한다.

적어도 우리는 구성주의적(Compositional terms)으로 세계를 이해하고 받아들인다. 우리가 새로운 것을 배운다면 기존의 지식 구조에 새로운 지식을 맞춰 넣거나, 기존의 지식 구조를 새로운 지식에 맞춰 조정하거나 둘 중 하나이다.

기존의 많은 머신러닝 기법들은 구조적인 접근 방법들이 대부분이었다. 이는 data와 computing resource가 상대적으로 매우 비쌌다는 현실적인 문제를 가지고 있었기 때문이다.

반대로 딥러닝은 End-to-End 디자인 철학이라고 불릴만큼 priori representation을 최소화하는 방법이다. explicit structure와 hand-engineering을 최소화하는 방법을 찾는다. 이는 반대로 데이터와 연산 비용이 크게 줄어들었기 때문이다.

딥러닝을 통해 많은 문제들을 해결할 수 있었지만 여전히 Combinatorial Generalization을 요구하는 복잡한 문제들은 해결하지 못하고 있다. 물론 Structure Based Method 또한 곧바로 이러한 문제들을 해결할 수 없다.

우리가 제안하는 방법은 두 가지 방법, 즉 딥러닝과 Structure-based Method를 함께 사용하는(hybrid) 것이다.

사실 기존의 방법들도 명시적이진 않더라도 relational assumption을 가정하고 있다.

[Relational Reasoning]

- Structure를 known building blocks를 구성하는 product로 정의한다.
- Structured Representations는 이러한 구성을 포착한다.
- Structured Computations는 element와 composition에 전체에 대해 수행한다.
- Relational Reasoning은 entities와 relations, rules가 어떻게 구성되어있는지 structured representation으로 표현한다.
- entity란 attribute의 element를 의미한다.
- relation이란 entitiy 간의 property를 의미한다.
- rule이란 entity와 relation을 다른 entity와 relation으로 매핑하는 함수를 의미한다.

[Inductive Bias]


Building Block의 대표적인 예로 MLP, CNN, RNN, GNN 등이 있다.

layer를 구성하는 것은 relational inductive bias의 특정 타입을 제공한다.

non-relational inductive bias도 있다. activation, weight decay, dropout, batchnorm 등이 있다. 논문에서는 relational inductive bias만 다룬다.
