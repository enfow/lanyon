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

AI가 인간의 지적 수준을 달성하기 위해서는 Combinatorial Generalization가 필수적이다.

Combinatorial Generalization이란 알고 있는(knowing) Building Block으로 새로운 추론, 예측 등을 구성하는 것을 말한다. 이를 통해 유한한 자원으로 무한한 성능을 내는 것이 가능해진다.

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

- Inductive Bias의 정의([Springer Link](<https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_927>))

```
In machine learning, the term inductive bias refers to a set of (explicit or implicit) assumptions made by a learning algorithm in order to perform induction, that is, to generalize a finite set of observation (training data) into a general model of the domain.
```

- 보지 못한 데이터에 대해서도 귀납적 추론을 하기 위해 모델(알고리즘)에서 가지고 있는 가정들의 집합. 선형회귀의 선형성 가정, 베이지안 모델에서 주어지는 Prior Distribution 등이 대표적이다.
- Regularization Term 또한 Inductive Bias 중 하나라고 할 수 있다.

Building Block의 대표적인 예로 MLP, CNN, RNN, GNN 등이 있다.

layer를 구성하는 것은 relational inductive bias의 특정 타입을 제공한다.

non-relational inductive bias도 있다. activation, weight decay, dropout, batchnorm 등이 있다. 논문에서는 relational inductive bias만 다룬다.
