---
layout: post
title: pyenv, virtualenv, autoenv
category_num : 3
keyword: '[Python]'
---

# pyenv, virtualenv, autoenv

- update at: 2020.02.02, 2020.08.18

## PYENV

[PYENV GITHUB](<https://github.com/pyenv/pyenv>)

pyenv는 한 시스템에 여러 개의 파이썬을 설치하고 사용자가 필요할 때마다 빠르게 파이썬 환경을 전환할 수 있도록 도와주는 도구이다.

### 0. Installation & settings

```
$
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
source ~/.profile
```

### 1. python installation

**python 설치 전 기본 환경 설정**

```
$
sudo apt-get install -y libsqlite3-dev zlib1g-dev libssl-dev libffi-dev libbz2-dev liblzma-dev
sudo apt-get install -y python3-setuptools python3-pip
```

파이썬 설치 전에 해주는 것이 좋다. 만약 파이썬 설치에 실패하면 다음 명령어도 실행한다.

```
$
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
```

**python 3.7.4 설치**

```
pyenv install 3.7.4
```

**설치 가능 python version 목록 확인**

```
pyenv install -list
```

### 2. check available python versions

**현재 사용 중인 파이썬 버전 확인**

```
pyenv version
```

- `system` 으로 확인되면 로컬 환경의 기본 파이썬으로 설정되어 있다는 것을 의미

**설치된 파이썬 버전 확인**

```
pyenv versions
```

- 현재 사용 중인 파이썬 버전은 `*`로 확인 가능

### 3. change python version

**파이썬 3.7.4 버전으로 파이썬 환경 전환**

```
pyenv shell 3.7.4
```

**글로벌 파이썬 환경 설정**

```
pyenv global 3.7.4
```

## VIRTUALENV

[PYENV-VIRTUALENV GITHUB](<https://github.com/pyenv/pyenv-virtualenv>)

**virtualenv**는 격리된 파이썬 환경을 만들어주는 도구라고 할 수 있다. 프로젝트에 따라 특수한 파이썬 패키지 및 환경 설정이 필요한 경우가 있는데 virtualenv를 통해 충돌을 방지하면서도 빠른 환경 전환이 가능하다. **pyenv-virtualenv**는 pyenv 플러그인으로 pyenv에서 virtualenv를 사용할 수 있도록 해주는 도구이다.

### 0. Installation & settings

pyenv-virtualenv를 사용하기 위해서는 pyenv를 먼저 설치해야 한다.

```
$
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
```

### 1. create virtuelenv

**파이썬 3.7.4 버전을 사용하는 virtualenv 환경 생성**

```
$
pyenv shell 3.7.4
pyenv virtualenv <virtualenv_name>
```

### 2. check virtualenv list

**설치된 virtualenv 목록 확인**

```
pyenv virtualenvs
```

### 3. activate virtualenv

**특정 virtualenv 환경 진입**

```
pyenv activate <virtualenv_name>
```

**현재 virtualenv 환경 해제**

```
pyenv deactivate
```

### 4. virtualenv remove

**virtualenv 삭제**

```
pyenv uninstall <virtualenv_name>
```

## AUTOENV

[AUTOENV GITHUB](<https://github.com/inishchith/autoenv>)

autoenv는 특정 디렉토리 진입 시 자동으로 환경 설정이 가능하도록 도와주는 기능을 제공한다.

### 0. Installation

```
$
git clone git://github.com/kennethreitz/autoenv.git ~/.autoenv
echo 'source ~/.autoenv/activate.sh' >> ~/.bash_profile
source ~/.bash_profile
```

### 1. Usage

autoenv는 `.env`파일을 사용하며, 해당 디렉토리에 `.env`파일이 있으면 파일의 스크립트를 자동으로 실행해주는 것으로 이해하면 쉽다.

**linux command 'touch'로 만들기**

```
touch .env
echo "pyenv activate <pyenv-virtualenv_name> > .env"
```

**vim으로 만들기**

- 아래 명령어로 .env 파일 진입 후 `pyenv activate <pyenv-virtualenv_name>` 저장

```
vim .env
```

**.env 적용하기**

- 현재 디렉토리로 재진입하면 `.env`의 적용 여부를 묻는 내용이 나온다. `y`를 입력하면 이후에는 autoenv가 적용된다.

```
cd ./
```
