---
layout: post
title: Docker Introduction
---

# Docker Introduction

## docker의 정의

- **"컨테이너 기반의 오픈소스 가상화 플랫폼"**
  - 동일한 인터페이스를 이용하여 프로그램과 실행환경을 '컨테이너'로 추상화하여 관리하게 해주어 프로그램의 배포 및 관리를 도와준다.
  - 백엔드, 데이터베이스 등 어떤 프로그램도 추상화가 가능하다.
- docker는 기본적으로 **리눅스 컨테이너** 기술이다. 따라서 mac, windows 등의 운영체제에서는 가상머신의 설치가 필요하다.
- [docker docu](<https://docs.docker.com/>)
- [docker download](<https://hub.docker.com/?overlay=onboarding>)

### 기존 가상화 기술과의 비교

- VMware, VirtualBox 등은 OS 가상화 기술로 쉽게 말해 호스트 OS 위에 게스트 OS를 얹어 사용하는 방식이다. 간단하게 사용할 수 있다는 장점이 있지만 다음과 같은 단점이 있다.
  - 새로운 컴퓨터에 설치하는 것과 동일한 방법으로 OS를 설치해야해 비용이 크다.
  - 실질적으로 하나의 컴퓨터에서 두 개의 OS가 동작하는 것이기 때문에 성능 면에서 손실이 있다.
- 이러한 문제를 해결하기 위해 도입된 것이 docker와 같은 프로세스 격리 기술이다.

## image와 container

### image와 container 정의

- image와 container는 docker에서 가장 기본적인 개념이다.
- image는 container를 실행하는 데에 필요한 파일과 설정값 등을 기록해둔 도면이다.
  - ubuntu image라고 하면 ubuntu를 실행하는 데에 필요한 모든 파일을 가지고 있어야 한다.
- container란 격리된 공간에서 프로세스가 동작하도록 하는 기술을 말한다.
  - docker에서 container는 image를 바탕으로 만든 실제 실행 환경이라고 이해하면 된다.
- 기본적으로 container는 프로세스이므로 실행할 내용이 남아있지 않으면 종료된다.

### image 사용 방법

- 새로운 환경에서 기존 환경을 사용하고 싶다면 기존 환경의 image를 다운로드하고 이를 바탕으로 conatainer를 만들면 된다.
- 동일한 image를 가지고 여러 개의 container를 생성할 수 있으며, container의 설정이 변경되거나 container가 삭제되더라도 image에는 기본적으로 영향이 미치지 않는다.
  - 즉 해당 conatiner에 대한 변경 사항은 해당 container 자신에 국한된다.

### image layer

- docker의 image는 여러 개의 layer로 구성된 file system과 유사하다.
- 즉, 기존 image에 대해 변경 및 추가, 삭제가 필요한 경우 새로운 image를 생성하는 것이 아니라 기존의 image에 새로운 layer를 추가하는 방식으로 해결할 수 있다.

### image와 container 간의 관계

- 쉽게 말해 image는 class로, container는 class를 가지고 만든 instance로 비유할 수 있다.

- image를 통해 container를 생성한다.
- container의 변경사항은 image에 영향을 미치지 않는다.
- 현재 container의 상태를 기록한 image를 새롭게 생성할 수 있다.
  - (정확하게 말하면 image를 새로 생성하는 것이 아니라 기존 image에 변경사항이 기록된 layer를 추가하는 것이다)
- container는 활성/비활성의 두 가지 상태를 가지고 있다.
  - container의 process가 종료되면 container는 자동적으로 비활성 상태가 된다.
  - 비활성 상태라고 해서 삭제되는 것은 아니며 언제든지 활성 상태로 되돌릴 수 있다. 즉, container 상태로도 저장이 가능하다.

#### 관련 용어 정리

- run : image를 바탕으로 새로운 container 생성하는 것을 말한다.
- commit : container를 바탕으로 새로운 image 생성하는 것을 말한다.
- attach : 활성 container에 접속하는 것을 말한다.
- detach : container를 활성 상태로 두고 나가는 것을 말한다.
- start : 비활성 container 활성화하는 것을 말한다.

## server와 client

- docker는 docker server와 docker client로 구성된다.
- terminal에서 docker 명령어를 입력하면 다음과 같은 순서로 동작한다.
  1. docker command 입력
  2. client가 server에 command 전송
  3. server가 command 실행
  4. 실행 결과를 client에 전송
  5. client는 terminal에 결과 출력

## docker hub와 docker repo

- git hub와 유사하게 docker 또한 생성한 image를 repository에 저장하고 다른 사람들과 공유할 수 있다.
- ubuntu와 같은 공식 어플리케이션 또한 docker hub에서 관리되고 있다.
- [docker hub](<https://hub.docker.com/>)

## 명령어 모음

### container 관련 명령어

#### run

- container 생성

  `$ docker run ubuntu:16.04`
  
- ubuntu:16.04 라는 이름을 가진 image를 이용하여 container를 생성하게 된다.
  - 이때 ubuntu:16.04라는 이름의 image가 없는 경우에는 해당 이름의 image를 다운로드(pull)한 뒤 이를 가지고 container를 생성한다.

  `$ docker run -it ubuntu:16.04 /bin/bash`
 
- -it option을 전달하면 container 생성 후 command 입력 모드로 설정된다.
- `/bin/bash` 와 같이 image 뒤에 command가 입력되면 container 생성 후 해당 command가 실행된다.
- 결과적으로 위와 같은 명령어를 입력하면 bash shell이 실행된 리눅스 환경이 docker container로 실행된다.
- 이때 exit 명령어를 통해 bash shell을 종료하면 더 이상 container의 process가 없으므로 conatiner도 종료된다.

#### ps

- container 목록 확인

  `$ docker ps`

- 현재 활성화된 conatiner가 출력된다.

  `$ docker ps -a`

- 현재 활성 + 비활성 모든 conatiner가 출력된다.

#### start

- 비활성 container의 활성화

  `$ docker start <container_name or container_id>`

#### stop

- 활성 container의 비활성화

  `$ docker stop <container_name or container_id>`

#### attach

- 활성 container에 접근

  `$ docker attach <container_name or container_id>`

#### detach

- 활성 container에서 나가기(활성 상태 유지)

  `control + p + q`

#### rm

- container 제거
  
  `$ docker rm <container_id>`

### image 관련 명령어

#### images

- 이미지 목록 확인

  `$ docker images`

#### pull

- 이미지 다운로드

  `$ docker pull ubuntu:16.04`

- docker hub에서 ubuntu 16.04 버전의 image를 docker hub에서 다운로드한다.

#### search

- image 검색

  `$ docker search mysql`

- mysql이라는 이름을 가진 image를 docker hub에서 검색해 그 결과를 알려준다.

## reference

- [초보를 위한 도커 안내서 - 도커란 무엇인가?](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)
- [Docker란 무엇인가?](<https://bcho.tistory.com/805>)
