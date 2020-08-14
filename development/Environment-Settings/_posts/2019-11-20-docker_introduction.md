---
layout: post
title: Docker Introduction
category_num : 2
keyword: '[Docker]'
---

# Docker Introduction

- [docker docu](<https://docs.docker.com/>)
- [docker download](<https://hub.docker.com/?overlay=onboarding>)

## 1. 컨테이너 기반 가상화 플랫폼

**컨테이너 기반의 오픈소스 가상화 플랫폼**으로 정의되는 **docker**는 프로그램과 실행환경을 묶어 독립적인 container로 관리할 수 있게 해 주는 기술이다. 프로그램이 원하는 대로 동작하게 하려면 적절한 실행 환경이 갖추어져야 하는데, 프로그램에 맞춰 실행 환경을 다시 설정하는 것은 매우 귀찮으면서도 실수하기 쉬운 작업이다. docker를 이용하면 작업하고자 하는 프로그램에 맞춰 실행환경을 시시때때로 변경이 가능하며, 동시에 여러 사람이 작업을 하고자 할 때에도 간단히 실행환경을 맞출 수 있다.

기본적으로 docker는 기본적으로 **리눅스 컨테이너** 기술이다. 따라서 mac, windows 등의 운영체제에서는 가상머신의 설치가 필요하다.

### 기존 가상화 기술과의 비교

### OS 가상화 기술

docker가 나오기 전에도 이와 비슷한 기술들이 있었는데, 대표적인 것이 VMware, VirtualBox와 같은 OS 가상화 기술이다. OS 가상화 기술은 쉽게 말해 호스트 OS 위에 가상의 게스트 OS를 얹어 사용하는 방식이다. 복수의 OS를 한 번에 사용할 수 있다는 장점이 있지만 다음과 같은 단점이 있다.

- 새로운 컴퓨터에 OS를 설치하는 것과 동일한 방법으로 OS를 설치해야해 비용이 크다.
- 실질적으로 하나의 컴퓨터에서 두 개의 OS가 동작하는 것이기 때문에 성능 면에서 손실이 있다.

쉽게 말해 하나의 컴퓨터로 두 개의 OS를 띄우는 구조기 때문에 새로운 환경을 사용하기 위해 소요되는 시간적, 공간적 비용이 크고, 성능 면에서도 손실이 크다는 것이다.

### 프로세스 격리 기술

docker docu에 따르면 docker는 주어진 host OS에서 독립적인 container를 동시에 실행할 수 있다고 한다. 특히 OS 가상화 기술처럼 새로운 hypervisor를 실행하지 않고, host OS의 커널을 곧바로 사용하기 때문에 성능 면에서도 크게 유리한 면이 있다는 점을 강조한다.

## 2. Docker Architecture

<img src="{{site.image_url}}/development/docker_architecture.png" style="width: 35em">

docker는 기본적으로 **client-server 구조**를 갖는다. Client는 Docker Host와 REST API를 이용해 통신하게 되는데, Docker Host에서 Client와 통신을 담당하는 부분을 **Docker daemon**이라고 한다. client로부터 명령을 받으면 docker daemon은 그에 맞춰 Container와 Image, Networks 등 docker objects를 관리한다.

- docker client : Docker Host에 접근하기 위해 개별 사용자가 사용한다.
- docker daemon : client와 통신하며 docker object의 관리하고 필요에 따라 다른 docker daemon과 소통하기도 한다.
- docker objects : Image, Container, Networks, Volumes, Plugins 등이 대표적이다.
- docker registries : Docker Image를 저장하는 공간이다. Docker Hub가 대표적이다.

기본적으로 복수의 사용자가 하나의 Docker Host에 접근할 수 있고, 하나의 사용자가 복수의 Docker Host에 접근하는 것도 가능하다.

### client-server 구조

참고로 Host OS의 terminal에서 어떤 docker 명령어를 입력하면 다음과 같은 순서로 동작한다.

1. docker 명령어 입력
2. docker client가 docker daemon에 명령어 전송
3. docker daemon이 명령어에 맞춰 object 실행 및 관리
4. 실행 결과를 docker daemon이 docker client에 전송
5. docker client가 terminal에 결과 출력

## 3. Docker object: image, container

### image와 container 정의

image와 container는 docker에서 가장 기본적인 개념이다. **image**는 container를 실행하는 데에 필요한 파일과 설정값 등을 기록해둔 **도면**이라고 할 수 있다. 예를 들어 ubuntu image라고 하면 ubuntu를 실행하는 데에 필요한 모든 파일을 가지고 있어야 한다.

**container**란 격리 **프로세스**를 말한다. docker에서 container는 image를 바탕으로 만든 실제 실행 환경이라고 이해하면 된다. 기본적으로 container는 프로세스이므로 실행할 내용이 남아있지 않으면 종료된다.

### image 사용 방법

기존에 만들어진 환경을 사용하고 싶다면 기존 환경의 image를 다운로드하면 된다. 예를 들어 새로운 ubuntu 환경을 생성하고 싶다면 docker hub에서 적절한 ubuntu version의 image를 다운로드 하는 것이다. docker iamge가 갖추어지면 이를 이용해 conatainer를 생성할 수 있다.

동일한 image를 가지고 여러 개의 container를 생성할 수 있으며, container의 설정이 변경되거나 container가 삭제되더라도 image에는 기본적으로 영향이 미치지 않는다. 즉 어떤 container에 대한 변경 사항은 해당 container에만 국한된다.

### image layer

docker의 image는 여러 개의 layer로 구성된 file system과 유사하다. 기존 image에 대해 변경, 추가 및 삭제가 필요한 경우 새로운 image를 생성하는 것이 아니라 기존의 image에 새로운 layer를 추가하는 방식으로 해결할 수 있다.

### image와 container 간의 관계

python에 비유하면 image는 class로, container는 class를 가지고 만든 instance 라고 할 수 있다. 구체적으로는 다음과 같은 특성을 갖는다.

1. image를 통해 container를 생성한다.
2. container의 변경사항은 image에 영향을 미치지 않는다.
3. 현재 container의 상태를 기록한 image를 새롭게 생성할 수 있다.
  - 정확하게 말하면 image를 새로 생성하는 것이 아니라 기존 image에 변경사항이 기록된 layer를 추가하는 것이다
4. container는 활성/비활성의 두 가지 상태를 가지고 있다.
  - container의 process가 종료되면 container는 자동적으로 비활성 상태가 된다.
  - 비활성 상태라고 해서 삭제되는 것은 아니며 언제든지 활성 상태로 되돌릴 수 있다. 즉, container 상태로도 저장이 가능하다.

### 관련 용어 정리

- run : image를 바탕으로 새로운 container 생성하는 것을 말한다.
- commit : container를 바탕으로 새로운 image 생성하는 것을 말한다.
- attach : 활성 container에 접속하는 것을 말한다.
- detach : container를 활성 상태로 두고 나가는 것을 말한다.
- start : 비활성 container 활성화하는 것을 말한다.

## 4. docker registry: docker hub, docker repo

- [docker hub](<https://hub.docker.com/>)

git hub와 유사하게 docker 또한 생성한 image를 repository에 저장하고 다른 사람들과 공유할 수 있다. ubuntu와 같은 공식 어플리케이션 또한 docker hub에서 관리되고 있다.

## 5. 기본적인 docker 명령어

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

#### rmi

- 이미지 삭제

  `$ docker rmi <image_name>:<image_version>`

### container 관련 명령어

#### run

- container 생성

  `$ docker run ubuntu:16.04`
  
- ubuntu:16.04 라는 이름을 가진 image를 이용하여 container를 생성하게 된다.
  - 이때 ubuntu:16.04라는 이름의 image가 없는 경우에는 해당 이름의 image를 다운로드(pull)한 뒤 이를 가지고 container를 생성한다.

  `$ docker run -it ubuntu:16.04 /bin/bash`

  `$ docker run -it ubuntu:16.04 bash`

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

#### diff

- image와 비교해 container의 변경사항 확인

  `$ docker diff <container_id>`
  
#### commit

- 현재 container를 image로 만들기

  `$ docker commit <container_id> <new_image_name>:<new_version>`

- 예를 들어 ubuntu에 git을 설치한 후 이를 ubuntu:git으로 저장하고 싶다면 다음과 같이 하면 된다.

  `$ docker commit <container_id> ubuntu:git`

- 참고로 container_id의 경우 모두 입력하지 않고 중복되지 않는 선에서 최소한만 전달해도 인식한다.

## reference

- [초보를 위한 도커 안내서 - 도커란 무엇인가?](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)
- [Docker란 무엇인가?](<https://bcho.tistory.com/805>)
