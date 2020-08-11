---
layout: post
title: "Balancing a Robotic Spot Welding Manufacturing Line: an Industrial Case Study"
category_num : 1
keyword: '[Automobile/Robotics]'
---

# 논문 제목 : Balancing a Robotic Spot Welding Manufacturing Line: an Industrial Case Study(작성 중)

- Thiago Cantos Lopes, Celso Gustavo Stall Sikora, Rafael Gobbi Molina, Daniel Schibelbain, Luiz Carlos de Abreu Rodrigues, Leandro Magatao
- 2017
- [논문 링크](<https://www.sciencedirect.com/science/article/abs/pii/S0377221717305180>)
- 2020.08.11 정리

## Summary

## Keywords

- Spot Welding: 전류에 의한 저항으로 발생하는 열을 통해 접촉 금속 표면을 붙이는 용접 방법을 말한다([wiki](<https://en.wikipedia.org/wiki/Spot_welding>)).
- Flow Shop: 제품이 어떠한 순서로 생산되어지는지 생산 과정에 대한 분류로 Flow Shop은 일련의 정해진 과정을 한 번씩 차례대로 거치는 방법을 말한다. 반면 Job Shop에서는 제품의 특성에 따라 서로 다른 과정을 거쳐 생산된다. 일반적으로 Flow Shop은 자동차와 같이 복잡하고 큰 제품을 생산하기 위해 사용되며 자동화가 쉽다는 점, 측정이 쉽다는 점, 최적화가 쉽다는 점 등의 장점을 가진다. 반면 유연성이 떨어지고, 초기 비용이 많이 발생하며, 생산 능력을 향상시키는 것이 어렵다는 단점을 갖는다([link](<https://blog.robotiq.com/job-shop-vs-flow-shop-can-robots-work-for-both>)).
- Workstation: 제품에 대해 특정한 task를 수행하는 단위, 공정을 말한다.
- Assembly Line Balancing Problem: 각 Workstation의 수행 시간을 비슷하게 맞추어 Cycle time을 최소화하는 문제를 말한다.

## Flow Shop

자동차는 매우 많은 부품들로 이뤄져 있으며 자동차를 만드는 것은 이러한 부품들을 이어붙이는 과정이라고 할 수 있다. 작업 효율을 높이기 위해 자동차 공정에서는 **Flow Shop** 방식으로 생산이 이뤄지며 각 제품은 특정 작업들이 수행되는 Workstation을 차례대로 거쳐 완성된다.

**Workstation**은 생산 단계, 공정으로 생각할 수 있으며 복수의 작업들로 구성되는데, 각 workstation에 어떠한 작업들이 얼마나 분배되는지에 따라 효율성이 결정된다고 할 수 있다. 일련의 Workstation이 빠짐없이 차례대로 수행되어야만 하는 Flow Shop의 특성상 어느 한 Workstation에서 작업이 끝나지 않으면 다른 Workstation의 작업이 먼저 끝났다 할지라도 기다려야 한다. 따라서 Workstation 간 처리 시간을 비슷하게 맞추는 것이 Cycle time을 줄이는 데에 중요한 요인이 된다.

**Line Balancing**은 Workstation 간에 작업을 적절하게 배분하여 Cycle time을 줄이는 것을 목표로 한다. Line balancing을 평가하는 대표적인 지표로는 라인 편성 효율 산출식이 있으며, 모든 공정의 처리 시간이 동일하면 100%가 된다.

$$
\text{line balancing efficiency} =
{w_1 + w_2 + ... + w_n \over n \cdot \max(w_1, w_2, ... w_n) } * 100
$$

이때 $$\max(w_1, w_2, ... w_n)$$, 즉 workstation 중 가장 오래 걸리는 것을 system's bottleneck이라고 한다.

### Automotive Process

Michalos에 따르면 자동차는 다음 네 단계를 거쳐 제작된다고 한다.

- Stamping: 각 부품의 형태를 찍어내는 과정
- Body Shop: 부품들을 이어붙여 차체를 만드는 과정
- Painting: 도색 과정
- Final Assembly: 최종 조립 과정

위의 네 단계 중 Body Shop, Final Assembly 단계에 조립 공정(assembly operation)이 몰려 있다고 한다.

#### Body Shop

Body Shop은 형태가 갖춰진 부품들을 받아 차체로 만드는 과정으로, body shop의 결과물을 도색 이전의 차체라 하여 **Body-in-White**라고 한다.
