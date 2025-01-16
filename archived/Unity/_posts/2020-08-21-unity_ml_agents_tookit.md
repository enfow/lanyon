---
layout: post
title: Unity ML-Agents Toolkit
keyword: '[Unity]'
category_num : 1
---

# Unity ML-Agents Toolkit

- update at 2020.08.21
- Unity 2018.4
- Python 3.7.4

- [ML-Agents Overview](<https://github.com/Unity-Technologies/ml-agents/blob/release_6_docs/docs/ML-Agents-Overview.md>)

## Introduction

The Unity Machine Learning Agents Toolkit (ML-Agents Toolkit) is an open-source project that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API.

## Components

이미지 - unity_ml_agent_components

### 1. Learning Environment

Unity Scene과 모든 Game Characters를 포함하는 개념

Unity Scene은 Agent가 보고, 행동하고, 학습하는 환경이 된다.

ML-Agent Toolkit의 `ML-Agents Unity SDK`(com.unity.ml-agents package)를 사용하면 agent, behavior 등을 정의하여 보다 쉽게 학습 환경을 구축할 수 있다.

The Learning Environment contains two Unity Components that help organize the Unity scene:

#### 1.1. Agents

Unity GameObject(Sence 상의 Character)에 붙어 관찰하고, Action을 수행하며, Reward를 결정하게 된다.

모든 Agent는 Behavior와 연결되어 있다.

#### 1.2. Behavior

각각의 Behavior는 고유의 `Behavior Name` 필드로 존재한다.

Behavior는 Agent로 부터 Observation과 Reward를 받고, 외부로부터 받은 Action을 Agent에 전달하는 함수라고 생각할 수 있다.

Behavior에는 세 가지 유형이 있다.

- `Learning`: 학습 과정에 있는 경우를 말한다.
- `Inference`: Learning에 의해 학습된 모델로 평가하는 경우를 말한다.
- `Heuristic`: hard-coding된 규칙들을 사용하는 경우를 말한다.

모든 Agent는 반드시 하나의 Behavior와 연결되어야 한다.

복수의 Agent가 하나의 Behavior와 연결될 수 있다. 

Behavior를 공유한다고 해서 모든 Agent가 동일한 Observation과 Action 값을 가지지는 않는다.

하나의 환경에 복수의 Agent와 Behavior가 존재할 수 있다.

### 2. Python Low-Level API

Learning Environment와 상호작용하는 Python API이다. 

Unity 밖에 있다.

### 3. External Communicator

Learning Environment 내에서 Python API와의 상호작용을 담당한다.

### 4. Python Trainers

Agent를 학습시키는 머신러닝 알고리즘이 있는 곳

## Python Low Level API

[Python API](<https://github.com/Unity-Technologies/ml-agents/blob/release_6_docs/docs/Python-API.md>)

Python Low Level API는 Unity Environment(mlagents_envs)와 직접적으로 상호작용할 수 있도록 해준다.

Unity Environment의 simulation loop를 통제한다.
