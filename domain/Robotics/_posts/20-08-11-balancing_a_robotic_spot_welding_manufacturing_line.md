---
layout: post
title: "Balancing a Robotic Spot Welding Manufacturing Line: an Industrial Case Study"
category_num : 1
keyword: '[Automobile/Robotics]'
---

# 논문 제목 : Balancing a Robotic Spot Welding Manufacturing Line: an Industrial Case Study

- Thiago Cantos Lopes, Celso Gustavo Stall Sikora, Rafael Gobbi Molina, Daniel Schibelbain, Luiz Carlos de Abreu Rodrigues, Leandro Magatao
- 2017
- [논문 링크](<https://www.sciencedirect.com/science/article/abs/pii/S0377221717305180>)
- 2020.08.11 정리

## Summary

- Flow Shop 구조를 가지는 자동차 공정의 특성상 각 workstation의 작업 시간을 비슷하게 맞추는 것이 Cycle time을 줄이는 데에 큰 영향을 미치게 된다.
- Body Shop 단계에서 각 부품들을 용접하는 작업이 많이 요구되는데 복수의 로봇팔이 동시에 작업하다보니 로봇팔 별 작업의 분배, 충돌 방지 문제가 발생한다. 무엇보다 어떤 방식으로 하면 얼마나 작업 시간을 줄일 수 있는지 예측하는 것이 어려워 Line Balnacing을 힘들게 하는 요인이 된다.

## Keywords

- **Spot Welding**: 전류에 의한 저항으로 발생하는 열을 통해 접촉 금속 표면을 붙이는 용접 방법을 말한다([wiki](<https://en.wikipedia.org/wiki/Spot_welding>)).
- **Flow Shop**: 제품이 어떠한 순서로 생산되어지는지 생산 과정에 대한 분류로 Flow Shop은 일련의 정해진 과정을 한 번씩 차례대로 거치는 방법을 말한다. 반면 Job Shop에서는 제품의 특성에 따라 서로 다른 과정을 거쳐 생산된다. 일반적으로 Flow Shop은 자동차와 같이 복잡하고 큰 제품을 생산하기 위해 사용되며 자동화가 쉽다는 점, 측정이 쉽다는 점, 최적화가 쉽다는 점 등의 장점을 가진다. 반면 유연성이 떨어지고, 초기 비용이 많이 발생하며, 생산 능력을 향상시키는 것이 어렵다는 단점을 갖는다([link](<https://blog.robotiq.com/job-shop-vs-flow-shop-can-robots-work-for-both>)).
- **Workstation**: 제품에 대해 특정한 task를 수행하는 단위, 공정을 말한다.
- **Assembly Line Balancing Problem**: 각 Workstation의 수행 시간을 비슷하게 맞추어 Cycle time을 최소화하는 문제를 말한다.

## Flow Shop

자동차는 매우 많은 부품들로 이뤄져 있으며 자동차를 만드는 것은 이러한 부품들을 이어붙이는 과정이라고 할 수 있다. 작업 효율을 높이기 위해 자동차 공정에서는 **Flow Shop** 방식으로 생산이 이뤄지며 각 제품은 특정 작업들이 수행되는 Workstation을 차례대로 거쳐 완성된다.

**Workstation**은 생산 단계, 공정으로 생각할 수 있으며 복수의 작업들로 구성되는데 이때 각 workstation에 어떠한 작업들이 얼마나 분배되는지에 따라 효율성이 결정된다. 일련의 Workstation이 빠짐없이 차례대로 수행되어야만 하는 Flow Shop의 특성상 어느 한 Workstation에서 작업이 끝나지 않으면 다른 Workstation의 작업이 먼저 끝났다 할지라도 다음 단계로 넘어가지 못하기 때문에 Workstation 간 처리 시간을 비슷하게 맞추는 것이 Cycle time을 줄이는 중요한 요인이 된다.

**Line Balancing**은 Workstation 간에 작업을 적절하게 배분하여 Cycle time을 줄이는 것을 목표로 한다. Line balancing을 평가하는 대표적인 지표로는 라인 편성 효율 산출식이 있으며, 모든 공정의 처리 시간이 동일하면 100%가 된다.

$$
\text{line balancing efficiency} =
{w_1 + w_2 + ... + w_n \over n \cdot \max(w_1, w_2, ... w_n) } * 100
$$

이때 $$\max(w_1, w_2, ... w_n)$$, 즉 workstation 중 가장 오래 걸리는 것을 system's bottleneck이라고 한다.

## Automotive Process

Michalos에 따르면 자동차는 다음 네 단계를 거쳐 제작된다고 한다.

- Stamping: 각 부품의 형태를 찍어내는 과정
- Body Shop: 부품들을 이어붙여 차체를 만드는 과정
- Painting: 도색 과정
- Final Assembly: 최종 조립 과정

위의 네 단계 중 Body Shop, Final Assembly 단계에 조립 공정(assembly operation)이 몰려 있다. 이 중 Body Shop은 Final Assembly에 비해 자동화가 많이 이뤄져 있다고 한다.

### Body Shop

Body Shop은 형태가 갖춰진 부품들을 받아 차체로 만드는 과정으로, body shop의 결과물을 도색 이전의 차체라 하여 **Body-in-White**라고 한다. Stamping 과정을 통해 기본적인 형태가 갖춰진 부품들은 Resistence Spot Welding 방법을 통해 서로 이어붙여지게 된다. 이때 두 부품을 용접하는 데에 소요되는 시간은 1초 내외로 매우 짧으나, 용접 작업을 수행하는 로봇팔들을 point까지 움직이는 데에 대부분의 시간이 소요된다. 따라서 로봇팔이 용접해야 할 point 들을 효율적으로 순회하는 것이 작업 효율에 큰 영향을 미친다.

한 가지 더 고려해야 할 사안이 있다면 하나의 Workstation에 복수의 로봇팔이 존재한다는 것이다. 따라서 각각의 로봇에 어떤 point를 어떻게 분배할 것인가의 문제가 있고, 동시에 로봇 간의 충돌을 방지해야 한다. 자동차는 크기에 따라 3000개에서 7000개의 용접 point를 가지고 있으므로 각각의 로봇팔이 최대한 빠르게 이들을 순회하면서 서로 충돌하지 않도록 하는 것이 Body Shop Line Balancing의 핵심이라고 할 수 있다.

### Spot Welding

**Resistence Spot Welding(RSW)** 방식에 대해 조금 더 정리하자면, Resistence Spot Welding는 말 그대로 어떤 물체에 전류를 흐르게 했을 때 발생하는 **저항(Resistence)**을 사용하여 용접하는 방식이다. RSW는 1초 내에 용접을 수행할 수 있다는 장점을 가지고 있으나, 전류를 흘려보내기 위해 붙이고자 하는 조각의 양쪽에 모두 접근해야 한다는 점, 붙이고자 하는 point의 특성에 따라 다양한 크기의 전류값(electrodes)과 건(gun)을 제공해야 한다는 점 등이 단점으로 꼽힌다.

용접 point는 크게 두 가지로 나눌 수 있는데 **Piece Jointing**은 두 개의 부품을 서로 이어붙이기 위해 용접이 필요한 point를 말한다. 두 개의 떨어져 있는 부품을 붙여야 하므로 각 부품의 위치를 잡아주는 엑츄에이터(actuator)가 필요하다. **Reinforcement Procedure**는 제품의 강도를 조절하기 위해 수행하는 용접 작업으로 엑츄에이터가 없어도 된다.

이러한 특성들을 고려해 볼 때 Spot Welding 작업은 로봇팔 간의 point를 분배하는 것 외에도 다음과 같이 추가적인 문제들을 가진다.

- **Accessibility**: point 별로 특성이 다르므로 모든 로봇팔에 의해 모든 point들이 용접 가능한 것은 아니다. 또한 로봇팔의 위치와 형태, 생산품의 형태에 따라 모든 point에 대해 항상 접근 가능한 것도 아니다.
- **Process time**: 용접 과정에서 용접 자체에 소요되는 시간은 매우 짧으나 point에 도달하기 위해 로봇팔을 이동시키고 자세를 잡는 데에 많은 시간이 소요된다.
- **Multi-operator**: 하나의 workstation에서 다수의 로봇팔이 동시에 작업하며 이들 간의 공간 경쟁(space competition) 문제를 해결해야 한다.

## Problem Statement

논문에서는 자동차 산업에서 balancing of robotic spot welding lines 최적화 문제와 관련하여 다음 9개의 제약 조건을 제시한다. 즉 최적화를 위해서는 아래의 제약 조건들을 고려해야 한다.

#### C0 Occurence Constraint

용접 처리되어야 하는 point들이 정해져 있다.

#### C1 Assignment Restrictions

로봇팔에 장착되어 있는 장치와 로봇팔의 위치에 따라 접근 가능한 point가 다르며, 이에 따라 로봇팔에 point가 할당되어야 한다.

#### C2 Robot-wise dependent parameters

로봇팔마다 속도, 크기, 장치 등이 모두 다르기 때문에 이에 맞춰 point를 할당해주어야 한다.

#### C3 Cycle time is decisively influenced by movement time

Cycle time을 계산하기 위해서는 용접 자체에 소요되는 시간 뿐만 아니라 로봇팔, 건(gun), 부품 등의 이동 시간을 모두 고려해야 한다.

#### C4 Multiple positions for welding 

point에 따라 다양한 방향으로 접근 가능한 경우가 있기 때문에 이들 중 하나를 선택해야 한다.

#### C5 Multi-manned stations with interference constraints

하나의 staion에 복수의 로봇팔이 동시에 작업을 수행하기 때문에 이들 간에 충돌 등을 고려해야 한다.

#### C6 Given welding parameters

로봇팔 별로 용접에 소요되는 시간과 접근 가능성은 주어져 있으므로 이를 고려해야 한다.

#### C7 Given layout

로봇팔의 특성(로봇팔의 수, 위치, 건(gun)의 형태 등)은 미리 정해져 있으며 변경이 불가능하다.

#### C8 No precedence relations

용접 point 간의 우선 관계는 없다. 즉 무엇을 먼저 용접하든 상관 없다.

### Cases

위에서 제시한 제약 사항과 관련하여 구체적인 사례들을 제시하고 있다. 그 내용은 다음과 같다.

#### 1) Accessible / Inaccessible

<img src="{{site.image_url}}/paper-review/welding_problem1.png" style="width: 35em">

용접 point마다 로봇팔의 특성에 따라 접근 가능한 곳도 있고 접근이 불가능한 곳도 있다.

#### 2) Moving time and Distance

<img src="{{site.image_url}}/paper-review/welding_problem2.png" style="width: 35em">

두 point의 거리는 매우 가까우나 로봇팔로 작업을 하기 위해서는 자세가 매우 크게 바뀐다. 즉, point 간의 거리 만으로는 로봇팔의 이동 시간을 유추할 수 없다.

#### 3) Multiple positions

<img src="{{site.image_url}}/paper-review/welding_problem3.png" style="width: 35em">

동일한 위치에 대해서도 서로 다른 자세로 접근이 가능하다.

#### 4) Collision

<img src="{{site.image_url}}/paper-review/welding_problem4.png" style="width: 35em">

다음과 같이 서로 작업하고자 하는 위치에 따라 로봇팔 간에 충돌이 발생할 수 있다.
