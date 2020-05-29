---
layout: post
title: (OS) CPU Scheduling
category_num: 4
---

# CPU Scheduling

- update date : 2020.05.28
- 본 포스팅은 고려대학교 컴퓨터학과 유혁 교수님의 2020년 1학기 Operating system을 수강하고 이를 바탕으로 작성했습니다. 수업 내용 복습을 목적으로 작성하였기 때문에 내용 중 부족한 점이 있을 수 있습니다.

## 1. Introduction

**CPU Scheduling**이란 말 그대로 CPU가 처리할 프로세스를 결정하는 것이다. CPU는 컴퓨터 하드웨어의 가장 중요한 구성요소로 컴퓨터 구조 설계 및 운영체제의 구현에 있어 핵심적인 고려사항이 된다. 컴퓨터 구조 이론에서는 컴퓨터의 성능을 평가하는 기준 중 하나로 CPU가 성능 향상의 병목(bottleneck)이 되는 것을 꼽는다. 즉 CPU 때문에 더 이상 성능을 높이지 못하는 상황이라면 주어진 상황에서 최적의 컴퓨터 구조라고 보는 것이다. 이와 비슷하게 운영체제에서는 CPU를 최대한 많이 사용하는 것을 성능의 중요한 기준으로 본다. 즉 CPU가 끊임없이 계속 프로세스를 처리하도록 하는 것이 운영체제 설계의 목표라고 할 수 있다.

## 2. Scheduling Criteria

CPU scheduling에 있어 고려대상이 되는 주요 기준은 다음과 같다.

|:------:|---|
|**CPU utiliztion**|CPU 사용률, 전체 시스템 시간 중 CPU가 작업을 처리하는 시간의 비율|
|**Throuhput**|처리량, 일정 시간 동안 완료되는 프로세스의 갯수|
|**Turnaround time**|개별 프로세스가 완료되는 데에 걸리는 총 시간|
|**Response time**|Input을 전달했을 때 그에 대한 Output을 받는 데 걸리는 시간|
|**Waiting time**|Ready Queue에서 프로세스가 기다리는 시간|

이상적인 CPU Scheduler 라면 짧은 시간 내에 최대한 많은 프로세스를 처리할 수 있도록 해야 한다. 따라서 CPU utiliztion, Throughput은 높으면 높을 수록 좋고 Turnaround time, response time, waiting time은 짧으면 짧을 수록 좋다.

어떻게 보면 기준이 매우 다양한데, 특히 시간과 관련된 기준이 많다. 이는 컴퓨터의 목적에 따라 주로 고려해야 하는 기준이 달라지기 때문이다. 채팅과 같이 Interactive한 작업에 대한 요구사항이 많은 개인용 PC에서는 Response time이 가장 중요한 기준이 된다. 반면 슈퍼 컴퓨터, 워크 스테이션과 같이 무겁고 오랫동안 처리해야 하는 프로세스를 다루는 컴퓨터의 경우 반응 속도는 조금 느리더라도 turn around time 등이 짧고, CPU 활용률을 높이는 것이 보다 효과적이다. 위의 모든 조건을 만족시키는 scheduler를 만드는 것은 사실상 불가능에 가깝기 때문에 사용자의 특성에 따라 scheduler의 구현 또한 달라지게 된다.

위의 기준을 충족시키기 위해 고려해야할 요소로는 대표적으로 아래와 같은 것들이 있다.

### Preemptive and Non-preemptive

Scheduling은 복수의 프로세스가 메모리 상에 올려져 있어 그 중 CPU가 작업할 프로세스를 고르는 것을 말하기 때문에 기본적으로 Multi-programming과 Time-sharing에서만 문제된다. 그렇다면 언제 Scheduling이 일어날까. 이와 관련하여 Operating system concepts에서는 다음 네 가지 상황을 언급하고 있다.

<img src="{{site.image_url}}/study/preemptive_and_non_preemptive.png" style="width:35em; display: block; margin: 0px auto;">

여기서 첫 번째와 네 번째는 I/O 또는 exit으로 인해 Process가 더 이상 진행될 수 없어 자진하여 CPU를 다른 프로세스에게 넘겨주는 경우이다. 즉 운영체제가 강제로 process로부터 CPU를 빼앗는 것이 아니다. 이러한 점에서 **Non-preemptive scheduling**(비선점형 스케쥴링)라고 표현한다. 

반면에 두 번째와 세 번째는 time interrupt 등으로 인하여 운영체제가 프로세스로부터 강제로 CPU를 빼앗았기 때문에 다음에 동작할 프로세스를 결정해야하는 상황이다. 이는 **preemptive scheduling**(선점형 스케쥴링)이라고 한다. 복수의 프로세스가 동작하고 있고 프로세스가 완료되기 전에 다른 프로세스가 처리를 시작할 수 있다는 점에서 preemptive scheduling에서는 동기화 문제가 존재한다.

Multi-programming과 Time-sharing의 차이는 프로세스가 스스로 내려오도록 기다리는지, 일정한 시간이 지나면 강제로 내리는지에 있다. 이러한 점에서 Multi-programming은 Non-preemptive scheduling만 사용한다고 할 수 있고, Time-sharing은 두 가지 경우 모두에서 Scheduling을 하게 된다.

### CPU burst and I/O burst

프로세스가 CPU 상에서 수행되고 있는 시간은 크게 **CPU burst**와 **I/O burst** 두 가지로 나누어지며 종료되기 전까지 필요에 따라 두 가지를 반복하게 된다. 

<img src="{{site.image_url}}/study/cpu_burst_io_burst.png" style="width:45em; display: block; margin: 0px auto;">

- CPU burst : CPU로 실제 연산을 수행하는 시간
- I/O burst : I/O를 기다리는 시간

상대적으로 입출력 횟수가 적고 연산량이 많은 프로세스의 경우 CPU burst가 길어지는데 이를 CPU-bound 프로세스라고 한다. 반대의 경우는 I/O bound 프로세스라고 한다.

그렇다면 CPU burst, I/O burst 가 CPU scheduling에 중요한 기준이 되는 이유는 무엇일까. 이는 아래 그림을 통해 확인할 수 있다.

<img src="{{site.image_url}}/study/cpu_burst_histogram.png" style="width:35em; display: block; margin: 0px auto;">

위의 그림을 보면 대부분의 CPU burst가 8ms 내에 완료된다는 것을 알 수 있다(시스템과 프로세스가 다르더라도 위와 같은 경향을 보인다고 한다). 즉 한 번 프로세스를 처리하기 시작하면 8ms 동안은 I/O가 발생하지 않아 계속 처리할 수 있다는 것이다. 이때 만약 Time Quantum을 8ms 보다 작게 가져간다면 context switch의 횟수가 크게 늘어날 것이고 그로 인한 오버헤드 또한 커질 것이라고 예상할 수 있다. 이러한 점에서 CPU burst의 경향은 Time quantum을 정하는데에 있어 중요한 기준이 된다.

## 3. Scheduling Algorithms

서론이 길었지만 CPU Scheduling은 결국에는 CPU가 비었을 때 Ready Queue에서 프로세스를 꺼내 CPU에게 전달해주는 것이다. 따라서 CPU Scheduler를 구현하는 것의 핵심은 Ready Queue에서 어떤 프로세스를 꺼내어 전달할 것인지 결정하는 것이다.

### 3.1. FCFS Scheduling

First-Come, First-Served, 한마디로 들어온 순서대로 보내겠다는 것이다. FIFO Queue로 쉽게 구현이 가능하다.

<img src="{{site.image_url}}/study/fcfs_scheduling.png" style="width:40em; display: block; margin: 0px auto;">

FCFS Scheduling는 가장 단순한 알고리즘으로 위의 그림에서도 확인할 수 있듯 Ready Queue 상의 프로세스 순서에 따라 Waiting time이 크게 바뀐다. 이러한 점에서 최적화의 가능성이 남아 있다고 할 수 있다.

### 3.2. SJF Scheduling

Shortest Jop First Scheduling은 CPU burst가 짧은 것을 먼저 처리하도록 하여 FCFS scheduling의 성능을 높이는 알고리즘이다. 평균 waiting time이 가장 짧다는 확실한 장점이 있다.

<img src="{{site.image_url}}/study/sgf_scheduling.png" style="width:40em; display: block; margin: 0px auto;">

SJF의 가장 큰 문제는 CPU burst의 길이는 Ready state에서는 알기 어렵다는 것이다. 이러한 문제를 해결하기 위해 과거의 CPU burst 길이를 가중평균하여 다음 CPU burst 길이를 예측하는 방식을 사용하기도 한다.

### 3.3. Priority Scheduling

Priority Scheduling은 말 그대로 프로세스마다 우선순위가 있어 우선순위가 높은 프로세스를 먼저 처리하는 방법이다. 우선순위는 대개 0부터 8 또는 0부터 4095와 같이 일정한 범위 내의 정수로 정해지는데 0이 가장 높은 것인지 낮은 것인지는 구현에 따라 다르다고 한다(CPU scheduling의 많은 부분들이 이와 같이 design choice로 남아 있다). 참고로 우선순위가 동일하면 FCFS 방식에 따라 먼저 들어온 프로세스를 우선 처리하게 된다.

Priority Scheduling은 `Starvation`이 발생할 가능성이 있다(루머에 따르면 priority scheduling에 따라 운영되던 MIT의 한 컴퓨터에서 1967년에 시작된 프로세스가 1973년에 발견되었다고 한다). 이러한 문제를 해결하는 방법으로는 **Aging**, 즉 기다리는 시간이 길어질수록 프로세스의 priority를 높여주는 방법이 있다.

### 3.4. Round Robin Scheduling

처음 보았던 FCFS scheduling은 Ready Queue의 순서대로 프로세스를 시작한다는 특징과 함께 프로세스가 종료될 때까지 계속 진행한다는 특징을 가지고 있었다. Round Robin은 Ready Queue의 순서대로 진행하는 것은 동일하지만 프로세스가 끝날 때까지 기다리는 것이 아니라 일정한 Time Quantum 이 지나면 강제로 프로세스를 내리고 다음 프로세스를 진행하도록 하는 방법이다. 이렇게 하여 FCFS의 단점이었던 순서에 따라 waiting time이 크게 바뀐다는 문제를 해결한다.

<img src="{{site.image_url}}/study/round_robin_scheduling.png" style="width:40em; display: block; margin: 0px auto;">

Round Robin Scheduling은 Time quantum을 얼마나 크게 설정하느냐에 따라 성능이 달라진다. Time quantum이 너무 짧으면 Context Switching으로 인한 오버헤드가 크게 늘어난다. 반대로 너무 길게 잡으면 FCFS와 동일해져 waiting time이 줄어드는 이점이 작아진다. 일반적으로는 전체 CPU burst 길이의 80% 수준으로 Time quantum을 설정한다고 한다.

### 3.5. Multi level Queue Scheduling

복수의 Ready Queue를 만들고 각각에 level을 두는 방법이다. 이러한 방식은 Priority Scheduling과 Round Robin Scheduling을 결합하여 만든 방식, 즉 우선 순위가 높은 것은 먼저 처리하되 우선 순위가 같은 경우에는 일정한 time quantum에 따라 번갈아 처리하는 방식을 개선하는 과정에서 만들어졌다고 한다. Priority Scheduling은 Priority가 가장 높은 프로세스를 찾기 위해 항상 전체 Queue를 탐색해야하는데 Round Robin Scheduling으로 인해 탐색 횟수가 크게 늘어난 것이 문제였고, 이를 해결하기 위해 탐색을 없애고 priority에 따라 Queue를 따로 두는 방식을 고안한 것이다.

<img src="{{site.image_url}}/study/multi_level_scheduling.png" style="width:35em; display: block; margin: 0px auto;">

여기서 재미있는 것은 각각의 Queue마다 Scheduling 방식을 다르게 설정할 수 있다는 것이다. 예를 들어 가장 높은 우선 순위를 갖는 Queue에서는 time quantum이 8ms인 Round Robin을, 두 번째 Queue에서는 16ms인 Round Robin을 하면서 마지막 Queue에서는 FCFS에 따라 Batch programming을 하는 것도 가능하다. 이렇게 하므로써 프로세스의 특성에 따라 대응력을 높일 수 있게 되는데, 예를 들어 real time 프로세스는 최상위 Queue에 넣어 I/O가 빠르게 이뤄질 수 있도록 하고, Deep learning과 같이 많은 연산량을 요구하는 프로세스는 하위 프로세스에 넣어 연산이 끊기지 않도록 하는 것이다.

문제는 Priority Scheduling과 마찬가지로 Multi level Queue Scheduling 에서도 Starvation 문제가 발생한다는 것이다. 즉 항상 상위 Queue를 먼저 처리하므로 하위 Queue의 프로세스들이 오랫동안 처리되지 못할 가능성이 있다.

### 3.6 Multi level Feedback Queue Scheduling

Priority Scheduling에서는 Aging을 도입하여 Starvation 문제를 해결하려 했었다. Multi level Queue Scheduling에서도 비슷한 접근 방법으로 priority를 수정하여 해결하려고 하는 시도가 있는데 이것이 바로 Queue 간의 이동을 가능하게 하는 Multi level Feedback Queue Scheduling 이다.

<img src="{{site.image_url}}/study/multi_level_feedback_scheduling.png" style="width:35em; display: block; margin: 0px auto;">

Multi level Queue Scheduling 에서는 priority에 따라 프로세스가 진입하는 Queue가 달랐지만 Multi level Feedback Queue Scheduling에서는 모두 최상위 Queue에서 시작하고 동일한 time quantum을 부여받는다. 이렇게 되면 동일한 시간 내에 완료되는 프로세스가 있는 반면 그렇지 못한 프로세스도 있을 것인데 아직 처리할 것이 남은 프로세스는 하위 Queue로 보내어 보다 긴 Time quantum을 가지고 처리하도록 한다. 여기서 각 level 별 Queue의 Scheduling 방식, downgrade 되는 규칙 등은 모두 정해진 것 없이 구현 과정에서 결정해야 하는 design choice이다. 
