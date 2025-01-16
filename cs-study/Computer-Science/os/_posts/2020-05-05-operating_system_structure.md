---
layout: post
title: Monolithic kernel and MicroKernel
category_num: 102
keyword: "[OS]"
---

# Monolithic Kernel and MicroKernel

- update date : 2020.05.05
- 본 포스팅은 고려대학교 컴퓨터학과 유혁 교수님의 2020년 1학기 Operating system을 수강하고, 이를 바탕으로 작성했습니다. 수업 내용 복습을 목적으로 작성하였기 때문에 내용 중 부족한 점이 있을 수 있습니다.
- Operating System Concepts 10th Edition 또한 참고했습니다.

## Kernel Structure

운영체제의 구조 혹은 커널의 구조는 각각의 기능들이 얼마나 강하게 묶여 있는가에 따라 **tightly coupled system**과 **loosely coupled system** 두 가지로 나누어진다. 쉽게 말하면 여러 기능을 가진 하나를 만들 것인지, 각각의 기능을 만들어 하나처럼 묶을 것인지의 문제라고 할 수 있다. 이때 커다란 하나의 커널을 만드는 것을 `Monolithic Structure` 라고 한다. 여러 개로 나누는 방법은 다양한데, 대표적인 것이 `Microkernels`이다.

## 1. Monolithic Structure

<img src="{{site.image_url}}/study/monolithic_1.png" style="width:35em">

`Monolithic`은 하나를 의미하는 `Mono`와 돌을 의미하는 `lithic`의 합성어이다. 구글에 lithic을 찾아보면 석기시대의 돌도끼 사진이 많이 나오는데, 당시 하나의 돌로 여러 종류의 작업을 했던 것처럼 **Monolithic은 하나의 kernel로 OS가 지원하는 모든 작업을 수행하는 것을 의미한다.** 즉 메모리 관리, CPU 스케쥴링 등 컴퓨터의 기본적인 운영을 비롯하여 입출력 장치 접근, 네트워크 장치 접근 등 커널이 수행하는 다양한 작업을 Monolithic Kernel에서는 단일의 메모리 공간에서 동작하게 된다.

Monolithic structure는 UNIX의 초기 버전에도 사용될 정도로 역사가 오래된 동시에, LINUX를 비롯하여 최근에도 가장 많이 사용되는 구조 중 하나이다. 이렇게 오랫동안 사용될 수 있는 것은 그 만큼 장점이 많기 때문이다. Monolithic Kernel의 가장 큰 장점은 속도가 빠르다는 점이다. Microkernels와의 비교에서 보다 확실히 드러나겠지만, Kernel이 수행해야 하는 모든 작업들이 Kernel 내부에서 수행되다보니 불필요한 Context switching을 줄일 수 있고, 외부로 작업 처리를 위한 message를 보낼 필요도 없어지기 때문에 상대적으로 오버헤드가 작다고 할 수 있다.

하나의 덩어리로 되어 있는 것은 반대로 유지 보수의 측면에서는 단점이 된다. LINUX에서는 system call을 추가하는 단순한 작업을 위해서도 kernel 전체를 다시 빌드해야 하는데, 시간이 매우 오래 걸린다. 즉 이처럼 커널을 조금만 변경하려고 해도 전체를 새로 만들어야 하기 때문에 시간이 많이 걸리고, 모든 기능들이 서로 얽혀있어 어떤 부분의 코드 변경이 예상치 못한 문제를 일으킬 수 있다는 문제가 있다.

## 2. Microkernels

`Microkernel`은 Monolithic kernel의 단점을 보완하기 위해 Kernel이 수행하는 작업들을 기능별로 나누어 모듈화하고 있다. 그리고 메모리 괸리, CPU 스케쥴링과 같이 운영체제가 수행하는 필수적인 작업들은 Monolithic Kernel과 마찬가지로 Kernel Mode에서 수행하지만 File system, device driver 등과 같이 그 이외의 작업들은 User Mode에서 이뤄지도록 하고 있다. 이렇게 **kernel의 필수적이지 않은 기능들을 User Mode에서 동작하는 개별 프로세스처럼 본다**는 점이 Microkernel의 가장 큰 특징이라고 할 수 있으며, 이때 Microkernel은 서버와 클라이언트의 관계처럼 통신하며 User Mode의 프로세스들을 관리하는 작업을 수행한다.

Kernel의 크기를 줄이는 것에 초점맞춰 개발된 Microkernel은 기존에 Monolithic kernel과 비교해 커널의 유지 보수가 쉽고 비교적 안정적이다라는 장점을 갖는다. 기능별로 쪼개어져 있으므로 특정 기능을 추가한다고 할 때 해당 모듈만 다시 만들면 되고, 특정 모듈에 문제가 생겨 멈추더라도 해당 모듈만 다시 띄우면 되기 때문이다. 하지만 반대로 Monolithic kernel의 장점이었던 속도는 Microkernel의 단점이 된다. 즉 Microkernel에서는 기존에 kernel mode에서 수행하는 작업들도 user mode에서 수행하기 때문에 context switch가 빈번하게 발생하고, 프로세스 간에 message 전달 과정에서 불필요한 작업이 추가되어 속도가 떨어진다는 문제가 있다.

참고로 Microkernel은 카네기 멜론 대학에서 1980년대 중반 Mach라는 이름으로 처음 공개되었고, 최근에는 Apple macOS, iOS의 기반이 되는 [Darwin](https://www.operating-system.org/betriebssystem/_english/bs-darwin.htm)으로 명맥을 이어오고 있다.
