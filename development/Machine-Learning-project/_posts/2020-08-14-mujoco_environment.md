---
layout: post
title: Mujoco Environment
category_num : 4
---

# Mujoco Environment

- update at 2020.08.14
- [Mujoco 홈페이지](<http://www.mujoco.org/>)

## Get License

mujoco를 사용하기 위해서는 라이센스가 필요하다. 라이센스는 mjkey.txt 라는 파일로 관리되는데 이를 받기 위해서는 [mujoco license](<https://www.roboti.us/license.html>) 페이지에서 **Account Number**와 **Computer Id**를 입력해야 한다.

<img src="{{site.image_url}}/study/mujoco_get_mjkey.png" style="width:40em; display: block;">

### Account Number

Account Number를 받기 위해서는 같은 페이지의 파란색 박스에 정보를 입력해야 한다.

<img src="{{site.image_url}}/study/mujoco_get_student.png" style="width:40em; display: block;">

학생용으로 하는 경우 무료로 1년 간 사용할 수 있는데 이 경우 메일 주소는 반드시 학생용 메일 계정으로 해야한다. 입력을 완료하고 `Request license` 버튼을 누르면 성공 여부를 알려주는 페이지로 이동하게 된다. Account Number는 입력한 메일 주소로 오게 되는데 곧바로 받을 수 있는 것은 아니고 최대 3일 정도 기다려야 한다.

### Computer id

Computer id를 구하기 위해서는 우선 `getid` 파일을 다운로드해야 한다. 이는 첫 번째 사진에서 mujoco를 설치할 컴퓨터의 운영체제에 맞게 `Win32`, `Win64`, `linux`, `OSX` 중 하나를 클릭하면 자동으로 다운로드 된다. 만약 서버에 설치하려 한다면 아래 명령어로 가능하다.

```
curl -o getid_linux https://www.roboti.us/getid/getid_linux
```

파일을 다운로드 받았을 때 아래와 같이 직접 뜯어보면 안 되고

```
vim getid_linux
```

아래와 같이 실행해야 한다.

```
./getid_linux
```

만약 permission error가 발생한다면

```
zsh: permission denied: ./getid_linux
```

아래 명령어로 권한을 다시 설정해주어야 한다.

```
chmod u+x getid_linux
```

이러한 과정을 거치면 아래와 같이 Computer id를 확인할 수 있다.

<img src="{{site.image_url}}/study/mujoco_get_computer_id.png" style="width:35em; display: block;">

### Get mjkey.txt

<img src="{{site.image_url}}/study/mujoco_get_mjkey.png" style="width:40em; display: block;">

위의 입력 정보를 모두 넣고 `Request Computer`를 하게 되면 mjkey.txt를 메일로 받을 수 있다.

## Install Mujoco

### download and unzip

<img src="{{site.image_url}}/study/mujoco_download.png" style="width:40em; display: block;">

위의 그림에서 사용하고자 하는 컴퓨터의 운영체제에 맞게 다운로드하면 된다. 여기서 숫자는 버전이라고 생각할 수 있다. 명령어로도 받을 수 있는데 linux의 경우 아래와 같다.

```
curl -o mujoco200.zip https://www.roboti.us/download/mujoco200_linux.zip
```

파일을 다운로드 받으면 zip 압축파일임을 알 수 있는데 아래 명령어로 압축을 풀면 된다.

```
unzip mujoco200.zip
```

### License Location

mujoco 라이센스를 관리하는 데에 사용되는 `mjkey.txt`파일은 

- `~/.mujoco/`
-  `~/.mujoco/mujoco200_linux/bin/` 

두 곳에 저장해야 한다. 라이센스가 제대로 적용되었는지 여부는 

- `~/.mujoco/mujoco200_linux/bin/simulate`

로 확인할 수 있다.

경우에 따라서는 아래와 같은 에러가 발생하는 경우가 있는데 rendering과 관련된 문제로 시각화만 되지 않을 뿐 환경 구축과는 전혀 관련이 없는 error로 볼 수 있다.

```
ERROR: could not initialize GLFW
```

### install mujoco-py

mujoco-py는 pip 로 간단하게 설치할 수 있다.

```
pip install mujoco-py
```

경우에 따라서는 error가 나기도 하는데 아래와 같이 clone을 통해 설치하는 것도 가능하다.

```
$ git clone https://github.com/openai/mujoco-py.git
$ cd mujoco-py
$ python3 setup.py install
```

mujoco-py가 mujoco 환경을 정확하게 잡기 위해서는 아래와 같은 환경변수 설정이 필요하다. `.bash_profile`에 추가해주면 된다.

```
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco200_linux/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200_linux/bin
```
