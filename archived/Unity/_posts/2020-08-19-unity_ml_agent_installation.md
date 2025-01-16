---
layout: post
title: Unity & ML-Agent Installation
keyword: '[Unity]'
category_num : 1
---

# Unity & ML-Agent Installation

- update at 2020.08.19
- Unity 2018.4
- Python 3.7.4

## Install Unity

Unity를 다운로드 하기 위해서 [Unity Download](<https://unity3d.com/kr/get-unity/download>) 페이지에서 `Unity Hub`를 다운로드 한다. Unity Hub는 Unity 버전 관리, 프로젝트 관리 등을 도와주는 관리 프로그램이라고 생각하면 된다.

### Get License

Unity Hub를 설치하고 실행하고 가장 먼저 해야하는 것은 Unity License를 만드는 것이다. Hub의 우측 상단에 있는 `Preferences` 아이콘을 클릭하고 `Lisence Management`로 이동하면 다음 페이지를 확인할 수 있다.

<img src="{{site.image_url}}/development/unity_get_lisence_page.png" style="width:35em; display: block; margin: 0em auto; margin-top: 2.5em; margin-bottom: 2.5em">

여기서 `ACTIVATE NEW LISENCE` 버튼을 누르면 라이센스를 발급하는 과정이 시작된다. 상업 목적으로 사용할 것이 아니라면 Unity Personal 라이센스로 무료 사용이 가능하다.

### Install

라이센스를 발급받았다면 Unity를 본격적으로 설치할 차례다. `Preference`에서 빠져나와 `Installs`로 이동한 뒤 `ADD` 버튼을 누르면 Unity 2018.4 등을 비롯하여 설치하고자 하는 Unity의 버전을 정할 수 있고, 다음 단계에서 실행 환경에 맞추어 모듈을 추가할 수 있게 되어있다. 설정이 모두 끝나면 기다리기만 하면 된다.

<img src="{{site.image_url}}/development/unity_install_page.png" style="width:35em; display: block; margin: 0em auto; margin-top: 2.5em; margin-bottom: 2.5em">

## Create ML-Agent Environment

Unity 설치를 완료하면 `Project`에서 새로운 프로젝트를 생성할 수 있다. `New` 버튼을 누르고 개발하고자 하는 환경을 선택하면 된다.

<img src="{{site.image_url}}/development/unity_create_project.png" style="width:35em; display: block; margin: 0em auto; margin-top: 2.5em; margin-bottom: 2.5em">

### Install ML-Agent Package

ML-Agent를 사용하기 위해서는 프로젝트에 ML-Agent Package를 설치해야 한다. 이를 위해 `Project Manager`를 다음과 같이 띄우고 ML Agent를 검색한 뒤 설치해준다.

<img src="{{site.image_url}}/development/install_unity_ml_agent_package.png" style="width:35em; display: block; margin: 0em auto; margin-top: 2.5em; margin-bottom: 2.5em">