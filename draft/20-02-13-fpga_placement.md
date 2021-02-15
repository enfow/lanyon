---
layout: post
title: Chip Placement on FPGA 프로젝트를 소개합니다!
category_num : 2
keyword: '[프로젝트 소개]'
---

# Chip Placement on FPGA 프로젝트를 소개합니다!

## 어떤 프로젝트인가?

MakinaRocks의 COP(Combinatorial Optimization Problem) 팀에서는 지난 2020년 9월부터 2021년 1월까지 반도체 설계 공정 중 하나인 Placement & Routing에 강화학습을 적용하는 프로젝트를 진행했습니다. AI Chip 설계를 전문으로 하는 펩리스(fabless) 스타트업 Furiosa AI와의 협업으로 진행되었으며, Furiosa AI가 가지고 있는 반도체 설계 기술과 Makinarocks의 산업 AI 역량을 결합하여 상용 FPGA EDA Tool과 비교해 효율적인 문제 해결의 가능성을 확인할 수 있었습니다. 

본 프로젝트는 지난 2020년 4월 Google에서 발표한 Chip Placement with Deep Reinforcement Learning 논문에 기초를 두고 있으며, 논문에서 제시하는 문제 정의를 참고하였습니다. 다만 ASIC을 대상으로 하는 논문과는 달리 FPGA를 대상으로 하였다는 점에서 큰 차이가 있습니다. 

## 문제 정의부터 살펴보기

우리가 일상적으로 사용하는 반도체는 모두 수백 수천만 개의 소자들로 구성되어 있으며, 각각의 소자들이 서로 전기적 신호를 주고 받도록 설계되어 있습니다. 이러한 반도체를 실제로 구현하기 위해서는 모든 소자들을 손톱 크기의 Chip Canvas에 배치하고 각각을 연결해주어야 합니다. 반도체 산업에서는 어떤 소자를 어디에 배치하고(Placement) 어떻게 연결할 것인지(Routing) 결정하는 공정을 Placement & Routing 공정, 줄여서 P&R 이라고 부릅니다.

COP 팀에서는 논리적으로 정의된 소자들의 연결 그래프를 입력으로 받아 개별 소자들의 물리적인 위치를 결정해주는 강화학습 에이전트를 만들어 P&R 공정, 그 중에서도 Placement 작업을 해결하고자 했습니다. Furiosa AI 측의 제안으로 FPGA(Field-programmable gate array)를 사용하여 진행하게 되었고, 관련 프로젝트를 처음 진행하는 만큼 최소 크기의 반도체 설계부터 시작했습니다.

문제 정의부터 반도체 산업의 특성이 많이 녹아져 있는 만큼, 문제를 이해하고 해결하는 모든 과정에서 관련 도메인 지식을 많이 요구하는 프로젝트였습니다. 문제 정의와 관련하여 구체적인 내용들을 FPGA, Netlist, Metric 세 가지로 나누어 정리해 보았습니다. 구체적인 설명에 들어가기에 앞서 FPGA는 바둑판이고 Netlist는 바둑돌이라는 느낌을 가지고 시작하시면 보다 쉽게 이해하실 수 있을 것 같습니다.

### FPGA란?

시스템 반도체의 일종인 FPGA는 Field Programmable Gate Array의 약자로, Programmable이라는 표현에서도 알 수 있듯이 사용자가 목적에 맞게 내부 로직을 변경할 수 있는 반도체를 말합니다. 일반적으로 사용하는 CPU, GPU의 경우 내부 구조가 모두 결정되어 있기 때문에 정해진 목적대로만 사용할 수 있습니다. 반면 FPGA는 사용자가 자신이 원하는 대로 내부 로직을 조작하는 것이 가능합니다.

따라서 FPGA를 사용하게 되면 강화학습 에이전트가 배치한 결과를 상용 EDA Tool에 입력으로 전달하여 성능을 평가하는 것 뿐만 아니라 실제 FPGA Board 상에서 구동이 가능한지 확인 가능합니다. ASIC을 사용한다면 실제 반도체를 제작하기 전까지는 실물을 대상으로 테스트를 진행하는 것이 불가능하다는 점에서 FPGA를 사용하게 되면 보다 효율적으로 검증 작업을 수행할 수 있습니다. 참고로 프로젝트에서 사용한 FPGA는 Xilinx 사의 U250 Board이고, EDA Tool은 동사에서 제작한 Vivado를 사용했습니다.

#### Programmable의 의미는?

FPGA에서의 배치 문제를 정확하게 이해하기 위해서는 FPGA의 동작 방식에 대해 간략히 알고 있어야 합니다. FPGA가 Programmable 한 이유는 모든 소자가 미리 배치되어 있고, 전선들 또한 개별 소자들을 서로 연결할 수 있도록 미리 배치되어 있기 때문입니다. 즉 FPGA에서 프로그래밍한다는 것은 어떤 소자를 활성화시킬지 결정하는 작업이라고 할 수 있습니다.

[보드 이미지]

이때 활성화의 단위 소자를 BEL이라고 부릅니다. 참고로 BEL은 FPGA를 구성하는 최소 단위이기도 한데, Xilinx FPGA는 다음과 같은 계층 구조로 되어 있습니다. 자세한 내용은 Xilinx의 [Rapid Wright](<https://www.rapidwright.io/docs/Xilinx_Architecture.html>)를 참고하시기 바랍니다.

[FPGA 계층 구조 이미지]

### 반도체 설계도, Netlist

Netlist는 반도체의 논리적인 설계도로서, 여기에는 반도체가 동작하려면 어떤 소자들이 필요하고, 각각의 소자들은 어떻게 연결되어 있는지 정의되어 있습니다. 이러한 Netlist에는 개별 소자들의 연결 관계만 담겨 있을 뿐 각각의 소자들의 위치 정보나 어떤 소자가 다른 소자와 얼마나 가까워야 하는지에 대한 정보는 포함되어 있지 않습니다. Chip Placement란 Netlist 설계도에 정의되어 있는 내용을 최적 동작이 가능하도록 Netlist를 구성하는 소자들의 위치를 결정하는 과정이라고 할 수 있습니다.

[배치 결과 txt 사진]

#### Macro & Standard Cell

Netlist를 구성하는 소자들은 그 크기 및 기능에 따라 Macro와 Standard Cell로 구분합니다. 상대적으로 크기가 큰 Macro는 RAM, DSP 등과 같이 그 자체만으로도 복잡한 기능을 수행할 수 있는 소자들입니다. 반면 Standard Cell은 LUT, Flip-Flop과 같이 단순한 연산이나 순간적인 데이터 저장 등의 기능만을 가지고 있습니다. 

Macro는 다시 내부 구조가 변경 가능한지 여부에 따라 Hard Macro와 Soft Macro로 나뉘는데, 현재 반도체 산업에서는 Hard Macro는 사람이 직접 배치하는 것이 일반적입니다. Google의 Chip Placement with Reinforcement Learning 논문에서는 사람이 배치하는 Hard Macro만 강화학습이 배치하도록 하고 있으며, 나머지 Stnadard Cell들은 Clustering 하여 Cluster 단위로 전통적인 알고리즘을 사용하여 배치합니다.

#### 최소 문제 정의하기

COP 팀에서 첫 번째 문제로 확보한 Netlist는 소자의 개수가 72개로 Macro가 2개, Standard Cell이 70개 였습니다. Google 논문에 따라 진행한다면 강화학습 Agent로 단 2개만 배치해야 하는 상황이었기 때문에 Agent의 성능을 평가하기에는 너무 적다는 의견이 있었습니다. 따라서 본 프로젝트에서는 72개 모두를 강화학습 Agent가 배치하도록 하여 실험을 진행했습니다.

### 평가는 무엇을 기준으로 하나?

반도체의 소자들을 연결하며 신호를 주고 받을 수 있게 하는 전선을 Wire라고 부릅니다. 이 Wire의 길이에 따라 반도체의 성능이 달라지고 경우에 따라서는 반도체가 정상적으로 동작하지 못하게 되기도 합니다. 이를 두고 반도체 산업에서는 P&R의 결과로 반도체 성능의 척도인 PPA(Performance, Power, Area)가 결정된다고 말합니다. 즉 개별 소자들을 어떻게 배치하느냐에 따라 각 소자들을 연결하는 Wire의 길이와 필요한 영역의 크기가 달라진다는 것입니다. 

이러한 점에서 P&R 결과의 평가 척도로 PPA를 보여주는 수치들을 주로 사용합니다. COP 팀 또한 이러한 수치들을 기준으로 모델의 최종 성능을 평가했습니다. 프로젝트에서 사용한 구체적인 모델 평가 지표는 다음과 같습니다.

- WNS(Worst Negative Setup-time Slack): Clock Frequency와 관련된 지표
- WHS(Worst Negative Hold-time Slack): Clock Frequency와 관련된 지표
- DP(Dynamic Power): Power와 관련된 지표
- RU(Routing Utilization Ratio): 반도체의 안정성과 관련된 지표

## 강화학습 환경 만들기

프로젝트에서 다룬 문제에 대한 간략한 소개에 이어 개발 과정에 대해서도 소개해보려 합니다. 어떤 문제에 강화학습을 적용하기 위해 가장 먼저 해야 하는 작업은 에이전트가 학습할 수 있도록 적절한 강화학습 환경(Environment)을 만드는 것입니다. 이때 환경으로 사용 가능한 시뮬레이터가 있다면 에이전트와 시뮬레이터를 연결하는 작업만 수행하면 되지만, 그렇지 못한 상황이라면 주어진 문제에 맞게 동작하는 환경을 직접 개발해야 합니다. 

FPGA와 관련해서는 Xilinx 사의 Vivado를 시뮬레이터로 사용할 수 있었습니다. 하지만 동작 속도가 느리고 에이전트와의 연결 과정에서 어려움이 예상되어 사용하지 않는 것으로 결론 내렸습니다. 대신 FPGA와 유사하게 동작하는 Python 프로그램을 개발하여 학습 환경으로 사용했습니다.

환경을 개발하는 과정에서 시작 단계에서는 예상치 못한 다양한 문제들을 경험할 수 있었습니다. Google 논문에서 모호하게 기술하고 있는 부분들을 실험과 추론을 통해 구체화하기도 하고, 안정적인 학습을 위한 속도 개선과 디버깅 작업에도 많은 시간을 소요했었습니다. 구체적으로 환경을 개발하면서 많이 고민한 이슈들로는 다음과 같은 것들이 있었습니다.

### (1) 어디에 배치할지 어떻게 결정할까

ASIC을 사용하는 Google 논문에서는 전체 Chip Canvas를 일정한 간격의 Grid로 나누고, 에이전트가 Action으로서 그 중 하나를 선택하도록 하고 있습니다. FPGA 또한 사용하고자 하는 Board에 동일한 방법으로 Grid를 적용할 수 있습니다. 다만 Chip Canvas 상에 다른 소자들과 겹치지만 않는다면 소자를 자유롭게 배치할 수 있는 ASIC과는 달리 FPGA는 각 소자의 타입에 따라 배치 가능한 위치가 미리 정해져 있다는 문제가 있었습니다.

이러한 FPGA의 본질적인 특성 때문에 Google 논문보다는 다소 복잡하게 Environment를 구성하게 되었습니다. 우선 에이전트가 소자를 배치할 Grid Cell을 선택하면 해당 Grid Cell 내에 포함된 BEL 중 임의로 하나를 추출하여 소자와 매핑하도록 하였습니다. Action에 적용되는 Masking 또한 타입에 따라 다르게 적용했다는 점에서도 소자 간 구분이 없어 하나의 Mask만 사용하는 논문과는 차이가 있습니다.

### (2) Reward는 어떻게 계산할까

Reward Function은 강화학습에서 가장 중요한 것 요소 중 하나로, 학습의 방향을 결정합니다. Google 논문에서는 아래와 같이 Reward Function을 제안하고 있습니다.

>$$
R_{p,q} = -\text{Wire Length}(p, g) - \lambda \text{Congestion}(p, g) \\
\text{S.t. } \text{density}(p,g) \leq \max_{\text{density}}
$$

여기서 Wire Length와 Placement Density는 어렵지 않게 해결할 수 있었습니다. Wire Length의 경우 배치된 Macro의 2차원 위치 정보를 통해 HPWL 방식으로 쉽게 구하는 것이 가능하고, Placement Density는 전체 FPGA Board 상에서 배치 영역을 적절하게 조절하는 것으로 제약 조건을 만족하도록 만들 수 있기 때문입니다. 다만 Routing Congestion은 FPGA의 내부 동작 방식을 비롯해 반도체 관련 지식이 필요한 만큼 Reward Function은 Furiosa AI에서 개발을 진행해주었습니다.

### (3) 환경에서 어떤 정보를 주어야 할까

Google 논문에서 제시하는 Observation의 유형으로는 Macro Feature, Netlist Graph, Current Macro id, Netlist Metadata, Mask 등이 있습니다. 그런데 각각의 정보들이 어떻게 구성되어 있는지에 대해서는 간략하게 예시 수준으로만 기술하고 있습니다. 이러한 모호성을 해결하기 위해 Reward를 예측하는 데에 도움이 되는지에 따라 정보를 추가해나가는 방식으로 다양한 실험을 진행했습니다. 최종적으로는 다음과 같은 정보들을 Observation으로 에이전트에 전달하도록 했습니다.

- Macro Feature: 타입(One-Hot), 위치(Row & Col), 포트 갯수
- Netlist Graph: 인접 매트릭스
- Current Macro id: Macro Feature에서의 Index
- Netlist Metadata: 사용하는 영역의 크기, Grid의 크기, Congestion Map, 총 Macro의 갯수, 총 Wire의 갯수
- Mask: 타입별 마스킹 정보

### (4) 동작 속도도 빠르게 해보자

강화학습에서 환경의 역할을 단순하게 정의하면 에이전트가 결정한 Action을 반영하고 그 결과인 Reward와 Next State를 에이전트에게 알려주는 것입니다. 강화학습 환경이 병목이 되지 않으려면 이러한 과정, 즉 Action을 받아 Reward와 Next State를 전달해주는 과정이 빠르게 이뤄져야 합니다. COP 팀에서는 이 과정이 1ms 이내로 처리되도록 목표를 잡았고, 실제 완성된 실험 환경에서는 0.1ms 수준에서 이뤄지고 있습니다.

## 강화학습 환경 디버깅하기

환경을 구축하고 첫 번째 에이전트 모델까지 개발을 완료하여 처음으로 학습을 시작한 이후 오랫동안 원하는 수준의 학습이 이뤄지지 못했습니다. 그 원인을 찾기 위해 불확실한 부분들을 정리하고 하나씩 검토하는 작업을 진행했습니다. 특히 환경과 관련해선 다음과 같은 부분들에 대해 집중적으로 검증했습니다.

- 환경 문제
    - Reward Function이 잘못되었을 가능성
    - 그 밖에 환경에 버그가 존재할 가능성

환경을 직접 개발하다보니 환경 구현과 관련하여 코드가 의도적으로 동작하지 않는 경우들이 디버깅 과정에서 발견되기도 하였습니다. 이러한 문제를 해결하기 위해 환경을 구성하는 모든 메서드에 대해 Unit Test 코드를 작성하였습니다. 또한 코드의 가독성과 유지보수성을 높이기 위해 꾸준히 리펙토링을 진행하기도 했습니다.

Reward Function 또한 직접 작성했기 때문에 유효성을 검증하는 작업이 필요했습니다. 이에 대해서는 최적 배치를 기준으로 임의성의 수준을 달리하며 구한 배치 결과들의 Reward를 비교하는 방식으로 확인하였습니다.

[Reward Function 검증 실험 이미지 - Wire length / routing congestion]

오른쪽으로 갈수록 임의성이 높은 배치의 결과를 보여줍니다. 이를 통해 임의성이 높아질수록 Reward를 구성하는 두 요소 Wire Length와 Routing Congestion 모두 증가함을 알 수 있습니다. 각각의 배치는 Vivado를 통해 확보한 8개의 서로 다른 조건에서의 최적 배치를 기준으로 일정 비율의 소자들의 위치를 임의로 변경하는 식으로 확보하였고, 각각의 값은 80개의 배치 결과를 평균하여 얻은 결과입니다.

## 에이전트 개발하기

에이전트는 State Representation을 추출하는 Feature Embedding과 직접적으로 Action을 결정하는 강화학습 Policy 두 영역으로 구성되어 있습니다. Feature Embedding은 Google 논문에 나온 내용을 바탕으로 직접 구현하였고, 강화학습 Policy로는 Ray의 Rllib을 사용했습니다.

### Feature Embedding

에이전트에서 가장 핵심적인 부분은 State를 적절하게 표현하여 강화학습 Policy가 쉽게 이해할 수 있도록 표현하는 부분이라고 생각합니다. 이를 위해서는 환경으로부터 받은 Observation을 적절하게 처리하여 State Representation으로 만들어주어야 합니다.

배치 대상이 되는 Netlist는 Node와 Edge로 구성되는 Graph 형태로 되어 있습니다. 따라서 좋은 State Representation을 확보히가 위해서는 Graph 데이터를 잘 처리할 수 있는 모델이 필요합니다.

>$$
\eqalign{
&e_{ij} = f c_1 (\text{concat}( f c_0(v_i)  \lvert f c_0(v_j) \lvert w^e_{ij}))\\
&v_i = \text{mean}_{j \in N(v_i)}(e_{ij})
}
$$

이와 관련하여 Google 논문에서는 위와 같은 구조를 제시하고 있으며, COP 팀에서는 논문의 내용을 최대한 따라하여 구현했습니다.

### 강화학습 알고리즘

강화학습 알고리즘으로는 Google 논문과 동일하게 PPO를 사용했습니다. 에이전트를 구현하면서 중요하게 고려한 점 중 하나는 학습이 되지 않을 때의 Search Space를 줄이는 것이었습니다. 프로젝트 수행 기간이 상대적으로 짧은 반면, 환경과 Feature Embedding을 직접 개발하디보니 불확실성을 최대한 줄일 필요가 있었습니다. 따라서 강화학습 알고리즘은 검증이 완료된 구현체를 사용하고 싶었고, 이러한 점에서 많은 사람들이 직접 사용하고 검증한 Ray의 Rllib을 사용하게 되었습니다.

## 그래서 얼마나 잘했나?

최종적인 배치 결과는 다음과 같습니다.

| Place-name | Dyn-power | WNS | WHS | Route-V | Route-H | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Placement | 0.014 | -0.494 | 0.124 | 0.0062914 | 0.00345094 | -2.744234 |
| Vivado Placement | 0.01 | 0.079 | 0.026 | 0.001188 | 0.001569 | -1.1707 |
| RL Placement | 0.01 | 0.163 | 0.046 | 0.00130047 | 0.00132467 | -1.053514 |

Random Placement는 모든 소자의 위치를 임의로 선택한 배치 결과를 말하고, Vivado Placement는 Vivado에서 찾은 최적 배치 결과를 말합니다. RL Placement가 COP 팀에서 개발한 모델의 배치 결과입니다. Overall Score는 각각의 Metric에 스케일에 따라 가중합한 결과로 높을수록 성능이 좋습니다. 정확한 계산식은 아래와 같습니다.

> Overall Score = WNS + WHS - 100 * Routing Utilization - 100 * Dynamic Power

Vivado Placement의 Overall Score가 -1.1707인 반면 강화학습 에이전트로 배치한 결과는 -1.053514로 나왔습니다. 이는 Random Placement 결과를 0으로, Vivado Placement 결과를 100으로 보았을 때 약 1.03에 해당하는 수치로 Vivado의 최적 배치와 비교해 볼 때 3% 정도 더 나은 배치 결과를 얻었다고 할 수 있습니다.

각 배치를 시각화하면 다음과 같습니다.

[Random 배치]

[Vivado 배치]

[RL 배치]

시각적으로 보더라도 COP 팀의 RL Placement가 가장 좁은 영역에 배치한 것을 확인할 수 있습니다. 참고로 파란색 사각형은 Vivado에서 배치한 결과를, 주황색 사각형은 제약조건으로 결정된 배치 결과를 의미합니다. RL Placement와 Random Placement의 배치 결과는 모두 Constraint로 Vivado에 전달되므로 모두 주황색 사각형으로 표현되고 있습니다. Routing은 모두 Vivado의 Solution을 따랐습니다.

## 문제정의부터 결과도출까지 함께하며 느낀 점

Google의 Chip Placement with Reinforcement Learning 논문은 강화학습 알고리즘을 적용하여 처음으로 Chip Placement 문제 해결을 시도한 논문입니다. 성능 및 상용화의 관점에서 볼 때 기존의 방법론들을 압도하지는 못하지만 강화학습 알고리즘을 현실 문제에 적용하여 유의미한 결과를 도출했다는 점에서 의미가 있습니다. 논문 발표 이후에 Google 뿐만 아니라 여러 연구 기관에서 후속 연구를 진행하고 있는 만큼 앞으로 후속 연구 및 관련 프로젝트 또한 지켜볼 필요가 있어 보입니다.

반도체 설계와 관련하여 전문적인 지식이 전무했지만 MakinaRocks COP 팀은 반도체 설계 전문 기업인 Furiosa AI와의 긴밀한 협력을 통해 반도체 설계 공정 상의 문제를 정의하고 결과를 도출할 수 있었습니다. 이 과정에서 머신러닝에서 사용되는 테크닉을 현실 문제에 적용하기 위해서는 도메인 지식과의 융화가 반드시 필요하다는 것을 다시 한 번 더 느낄 수 있었습니다.

나아가 강화학습과 관련하여 프로젝트를 진행하며 다음과 같은 인사이트를 느낄 수 있었습니다.

- 현실 문제를 해결하기 위해서는 State와 Reward Function을 어떻게 설정하느냐가 매우 중요하다.
- 전통적인 방식과 결합하여 강화학습을 적용하면 더욱 높은 성능을 확보할 수 있다.

강화학습을 공부한다고 하면 PPO, DQN, DDPG와 같은 학습 알고리즘에 집중하는 경향이 있는 것 같습니다. 물론 이러한 학습 알고리즘의 특성을 이해하는 것도 중요하지만 State Representation을 형성하는 방법이나 Reward Function을 설계하는 방법에 따라 전체적인 성능이 크게 달라지는 만큼 현실 문제에 강화학습을 적용하기 위해서는 이에 대한 정확한 이해가 필요하다고 생각합니다.

또한 전체 문제를 강화학습으로 푸는 것이 아니라 강화학습과 문제의 전통적인 알고리즘들이 가지는 특성을 정확히 이해하고 필요에 따라 분업이 이뤄져야 할 것으로 보입니다. 이와 관련해서는 단순히 성능 뿐만 아니라 연산에 소요되는 시간, 알고리즘의 범용성, Re-Training의 필요성 등이 주요 고려 요소가 될 것입니다.

마지막으로 위의 두 가지 모두 해결하고자 하는 문제의 특성을 정확히 알아야 가능한 부분이라는 공통점을 가지고 있습니다. 이러한 점에서 실험실이 아닌 현실 문제를 해결하기 위해서는 강화학습에서도 Domain Knowledge에 대한 깊은 이해와 머신러닝 지식에 대한 결합이 중요하게 여겨져야 할 것입니다.
