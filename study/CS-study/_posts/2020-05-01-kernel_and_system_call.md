---
layout: post
title: Kernel and System call
category_num: 1
keyword: '[OS]'
---

# Kernel and System Call

- update date : 2020.05.01
- 본 포스팅은 고려대학교 컴퓨터학과 유혁 교수님의 2020년 1학기 Operating system을 수강하고, 이를 바탕으로 작성했습니다. 수업 내용 복습을 목적으로 작성하였기 때문에 내용 중 부족한 점이 있을 수 있습니다.

## Introduction: Kernel

<img src="{{site.image_url}}/study/kernel_location.png" style="width:35em; display: block; margin: 0px auto;">

Kernel은 개념적으로 사용자의 User Application 과 컴퓨터의 하드웨어 사이에 위치한다. 그리고 두 가지를 이어주는 과정에서 하드웨어를 보다 효율적으로 사용하고, 여러 user application 이 원할하게 동작할 수 있도록 관리하는 역할을 수행한다. File system이나 I/O system과 같이 데이터를 다루는 것부터 시작하여 CPU Scheduling, Virtual Memory 등 하드웨어 자원을 보다 효율적으로 사용하기 위한 방법들이 모두 kernel에서 이뤄진다. 이러한 점에서 kernel은 운영체제의 가장 핵심적인 부분이라고 할 수 있다.

**kernel이 수행하는 역할 중 하나는 User Application이 필요로 하는 작업을 대신해주는 것이다.** 일반적으로 User Application은 대부분의 하드웨어 자원에 접근할 수 없고, kernel을 통해서만 가능하다. 예를 들어 카카오톡으로 메시지를 받았다고 할 때 카카오톡 어플리케이션이 직접 네트워크 장치로부터 메시지를 가져오는 것이 아니다. 카카오톡의 요청을 받은 kernel이 네트워크 장치에서 메시지를 가져와 전달해 주는 것으로 보아야 한다.

이렇게 User application이 하드웨어에 직접 접근하지 못하고 Kernel을 통해서만 가능하도록 한 것은 안정적으로 다수의 프로그램을 처리하기 위함이다. Multi programming 또는 Time sharing 에서는 복수의 프로그램이 동시에 수행된다. 이러한 상황에서 모든 프로그램이 CPU, Memory, I/O 등에 제한 없이 접근이 가능하다면 다른 프로그램이 작업을 수행하고 있는 영역을 조작하거나, 컴퓨터 시스템 운영에 필수적인 부분을 임의로 수정하는 것이 가능해진다. 이러한 문제를 막기 위해 User application은 제한적인 하드웨어 자원을 할당받아 작업을 수행하고 다른 하드웨어에 접근할 때에는 이를 전체적으로 관리하는 kernel의 허락을 받도록 한 것이다.

## Kernel Mode & User Mode

- `Kernel Mode` : 운영체제가 실행되는 상태, 모든 하드웨어 자원에 접근할 수 있는 상태
- `User Mode` : 사용자 프로그램이 실행되는 상태, 제한된 하드웨어 자원에만 접근할 수 있는 상태

이때 User application이 동작하는 상태를 `User Mode`라고 하고, Kernel이 작업을 수행하는 상태를 `Kernel Mode`라고 한다. 각 Mode에 따라 하드웨어에 대한 접근과 사용 가능한 명령어에도 차이가 있는데, Kernel Mode는 모든 하드웨어 장치에 대한 접근 권한을 가지고 있다. 반면 User Mode는 특정 User application을 수행하는 만큼 Kernel Mode에 비해 다소 제한된 권한을 가지고 있다.

이렇게 Mode에 따라 권한이 다르게 부여되고, 접근 가능한 장치 및 명령어가 제한된다는 것은 하드웨어 상에서도 확인이 가능한데, CPU의 Mode bit 이다. 두 가지 Mode를 나누는 것은 컴퓨터의 안정성을 위해 필수적인 만큼 CPU에서 하드웨어 적으로 사용할 수 있는 명령어를 제한하는 것이다. 참고로 CPU 제조사에 따라 mode bit의 크기도 다양한데 Intel의 경우 4 종류를, AMD의 경우 2 종류를 지원한다고 한다.

## SYSTEM CALL

**System call은 User application에서 Kernel에 작업을 요청할 때 사용하는 인터페이스**라고 할 수 있다. System call 이 수행되는 과정, 즉 `system call routine`은 다음과 같이 정리할 수 있다.

1. User application이 System call을 호출한다.
2. User mode에서 Kernel mode로 전환한다.
3. System call Interface가 전달받은 System call에 맞는 index로 System call table을 탐색한다.
4. System call table에서 해당 index에 매핑된 함수를 함께 전달받은 parameter와 함께 실행한다.
5. 함수의 결과를 User Application으로 전달한다.
6. Kernel mode에서 User mode로 전환한다.

<img src="{{site.image_url}}/study/system_call.png" style="width:40em; display: block; margin: 0px auto;">

위의 그림은 Operating System Concept 를 참조하여 그렸다. 그림에는 routine 에는 등장하지 않는 것이 두 가지 있는데, libc.a와 IDT이다. 먼저 **libc.a** 는 C언어 표준 라이브러리로, C언어에서 기본적으로 사용할 수 있는 함수를 포함한다. 기본적인 C 언어 프로그램을 작성할 때 system call에 관한 내용을 작성하지 않아도 되는 이유는 사용하는 C 언어의 기본 함수 정의의 이미 system call과 관련된 내용이 모두 포함되어 있기 때문이다.

**IDT**는 Interrupt Descriptor Table의 약자로, 인터럽트가 발생했을 때 어떤 작업을 수행할 것인지 알려주는 table이라고 할 수 있다. system call 또한 인터럽트의 일종이라고 할 수 있으며, IDT에서 0x80 index를 가지고 있다. 위의 그림에서 system call을 호출할 때 0x80를 함께 전달하는 것을 확인할 수 있다.
