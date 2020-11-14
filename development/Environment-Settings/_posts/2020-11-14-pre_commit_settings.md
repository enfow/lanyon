---
layout: post
title: Git Pre-Commit Settings
category_num : 8
keyword: '[Git]'
---

# Git Pre-Commit Settings

- Update Date: 2020.11.14
- Environment: MacOS 10.15.6 \| Python 3.7.5
- Reference: [Pre-Commit](<https://pre-commit.com/>) \| [black](<https://github.com/psf/black>) \| [isort](<https://github.com/PyCQA/isort>) \| [flake8](<https://flake8.pycqa.org/en/latest/>) \| [pylint](<https://pypi.org/project/pylint/>) \| [mypy](<http://mypy-lang.org/>) \| [pytest](<https://docs.pytest.org/en/stable/>)
- [Git Repo](<https://github.com/enfow/pre-commit-settings>)

**Pre-Commit**이란 Commit을 하기 전에 작성된 코드를 검사하여 문제가 있는지 확인해보는 것을 말한다. 이러한 기능을 활용하면 Linting, Formatting을 일괄적으로 적용할 수 있어 코드의 품질을 일정 수준 이상으로 유지하도록 할 수 있다는 장점이 있다. 본 포스팅에서는 Formatting과 Linting을 Pre-Commit에 적용하여 두 가지를 해결하지 못하면 Commit을 막는 방법을 다루고자 한다. 

참고로 Formatter와 Lintter는 버전에 따라 그 결과가 다를 수 있다. 따라서 설치 버전을 `requirements.txt`에 구체적으로 명시하여 공동 작업자 간에 버전을 맞추는 것이 중요하다. 포스팅을 위해 작성한 requirements.txt 파일은 다음과 같다.

```bash
# requirement.txt
# Formatter
black == 20.8b1
isort == 5.6.4

# Linter
flake8 == 3.8.4
flake8-annotations == 2.4.1
flake8-bugbear == 20.1.4
flake8-builtins == 1.5.3
flake8-docstrings == 1.5.0
flake8-mutable == 1.2.0
pylint == 2.6.0
mypy == 0.790

# pytest
pytest == 6.1.2
pytest-flake8 == 1.0.6
pytest-mypy == 0.8.0
pytest-pylint == 0.18.0
```

## 1. Formatter 

Formatter로는 [black](<https://github.com/psf/black>)과 [isort](<https://github.com/PyCQA/isort>)를 적용하고자 한다. black은 Python의 대표적인 Formatter로 PEP8 표준에 맞추어 작성된 코드를 변경해주고, isort는 import 문을 자동으로 정렬하여 준다. 두 가지 모두 `pip install`로 쉽게 설치가 가능하다.

```python
pip install black
pip install isort
```

사용 방식은 다음과 같다.

```
black <file/dir name>
isort <file/dir name>
```

각각에 대해 설정 값을 변경하여 Formatting 방식을 커스터마이징 하는 것도 가능하다. isort의 경우 `isort.cfg`라는 파일에서 설정이 가능하며, 구체적인 설정 방법은 [isort/options](<https://pycqa.github.io/isort/docs/configuration/options/>)에 나와있다. 참고로 black과 isort 모두 스크립트를 변경하기 때문에 두 가지가 동일한 룰에 따라 동작하는 것이 중요하다. isort에서는 다음과 같이 `isort.cfg`에서 [profile](<https://pycqa.github.io/isort/docs/configuration/profiles/>)이라는 설정 값으로 black과 동일한 규칙을 따르도록 할 수 있다.

```bash
#isort.cfg
profile=black
```

예를 들어 다음과 같이 작성된 파일이 있다면

```python
# test.py
1 import numpy as np
2
3 from typing import Tuple, Dict, Any, NamedTuple
4
5 import pandas as pd
6 import os
7
8 import matplotlib.pyplot as plt
```

isort를 적용한 결과는 다음과 같다.

```python
# test.py
1 import os
2 from typing import Any, Dict, NamedTuple, Tuple
3
4 import matplotlib.pyplot as plt
5 import numpy as np
6 import pandas as pd
```

## 2. Linter

Linter로는 [flake8](<https://flake8.pycqa.org/en/latest/>), [pylint](<https://pypi.org/project/pylint/>), [mypy](<http://mypy-lang.org/>)를 적용하려한다. formatter와 마찬가지로 모두 `pip linstall`로 쉽게 설치가 가능하다.

```bash
pip install flake8
pip install pylint
pip install mypy
```

각각을 사용하는 방법 또한 다음과 같이 매우 직관적이다.

```bash
flake8 <file/dir name>
pylint <file/dir name>
mypy <file/dir name>
```

### flake8

flake8의 가장 큰 특성 중 하나는 Plug-in을 추가로 설치하여 잡아낼 수 있는 문제를 사용자가 자유롭게 설정할 수 있다는 것이다. flake8의 대표적인 Plug-in은 [awesome-flake8-extensions repo](<https://github.com/DmytroLitvinov/awesome-flake8-extensions>)에서 확인할 수 있다. 포스팅을 위해 설치한 Plug-in은 다음과 같다.

- flake8-annotations
- flake8-bugbear
- flake8-builtins
- flake8-docstrings
- flake8-mutable
- pep8-naming

flake8의 configuration은 `.flake8`에서 다음과 같이 설정할 수 있다. 아래 예시는 [configuring flake8](<https://flake8.pycqa.org/en/latest/user/configuration.html>)에서 가지고 왔다. **ignore**로 특정 Error Code를 가지는 문제를 무시하도록 할 수 있고, **exclude**를 통해 파일 또는 디렉토리 단위로 무시할 수도 있다. **max-complexity**는 함수의 라인 수를 제한한다.

```bash
#.flake8
[flake8]
ignore = D203
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
max-complexity = 10
```

### pylint

flake8이 Plugin으로 기능을 확장할 수 있다면 **pylint**는 추가 설치 없이도 많은 문제를 잡을 수 있도록 기본적으로 많은 기능을 제공한다. [pylint](<https://pypi.org/project/pylint/>) 홈페이지의 소개 문구는 다음과 같다.

- Pylint is a Python static code analysis tool which looks for programming errors, helps enforcing a coding standard, sniffs for code smells and offers simple refactoring suggestions.

pylint의 설정 파일은 다음 명령어로 생성할 수 있다

```bash
pylint --generate-rcfile > pylintrc
```

설정 파일에서 disable 값으로 Error key value를 전달하면 특정 문제를 잡아내는 것을 제한할 수 있다.

```bash
# pylintrc
disable=print-statement,
        parameter-unpacking,
        unpacking-in-except,
        old-raise-syntax,
        backtick,
        long-suffix,
        old-ne-operator,
        ...
```

### mypy

[mypy](<>) 홈페이지에서는 첫 문장에서 mypy를 type checker로 소개하고 있다. 즉 mypy는 type과 관련된 문제에 특화된 linter로, 이를 위해 **type annotation**을 강제한다는 특성을 갖는다. 참고로 type annotation에 대한 Python 규정은 [PEP484](<https://www.python.org/dev/peps/pep-0484/>)에서 확인할 수 있다.

mypy의 설정 파일은 `mypy.ini`이고, 홈페이지에서 보여주는 예시는 다음과 같다.

```bash
# mypy.ini
# Global options
[mypy]
python_version = 2.7
warn_return_any = True
warn_unused_configs = True

# Per-module options:
[mypy-mycode.foo.*]
disallow_untyped_defs = True

[mypy-mycode.bar]
warn_return_any = False

[mypy-somelibrary]
ignore_missing_imports = True
```

## 3. Makefile

formatter와 linter를 5개 설치하였으므로 한 번 commit을 하기 위해서는 다섯 번의 명령어를 입력하여 하나씩 그 결과를 확인해야 한다. 번거로움을 줄이기 위해 `Makefile`을 작성하여 쉽게 그 결과를 확인하는 것도 가능하다.

```bash
# Makefile
format:
	black .
	isort .
```

이 경우 `make format` 명령어로 black과 isort 두 가지의 실행 결과를 한 번에 확인할 수 있다. 

```bash
lint
    flake8 .
    pylint .
    mypy .
```

그런데 linter의 경우 위와 같은 방식으로 하게 되면 flake8에서 문제가 검출되면 `pylint .`와 `mypy .`의 실행 결과를 확인할 수 없다. 모든 linter가 잡아낸 문제를 한 번에 확인하기 위해서는 `pytest`를 사용해야 한다.

### pytest

[pytest](<https://docs.pytest.org/en/stable/>)는 'The pytest framework makes it easy to write **small tests**'라고 명시하고 있듯이 Unit Test를 위해 자주 사용되는 framework이다. 여기서는 Unit Test는 고려하지 않고, 한 번에 여러 linter를 사용하기 위한 수단으로만 사용할 것이다. 설치 방식과 사용 방식은 위의 다른 도구들과 크게 다르지 않다.

```bash
pip install pytest
```

```bash
pytest .
```

pytest에서 flake8, pylint, mypy를 함께 사용하기 위해서는 다음을 추가적으로 설치해주어야 한다.

```bash
pip install pytest-flake8
pip install pytest-pylint
pip install pytest-mypy
```

사용 방식은 다음과 같다.

```bash
env PYTHONPATH=. pytest --flake8 --pylint --mypy --ignore=<ignore file/dir> --ignore=<ignore file/dir>
```

### Makefile Example

위와 같이 pytest를 적용하여 만든 Makefile은 다음과 같다.

```bash
# Makefile
format:
	black .
	isort .

lint:
	env PYTHONPATH=. pytest --flake8 --pylint --mypy
```

## 4. Pre-Commit

pre-commit을 수행한다는 것은 git의 hook으로 commit을 실시할 때마다 실행할 명령어들을 설정해두고, 이들 명령어들을 모두 통과하여야만 commit 프로세스를 진행하도록 하는 것이다. Github의 Repository를 `git clone` 받거나 `git init`으로 local repository를 만들게 되면 `.git/` 디렉토리가 생성되는데, 모든 hook들은 `.git/hooks`에 저장되어 있다. 따라서 pre-commit hook 또한 여기에 추가하는 방식으로 이뤄진다.

pre-commit hook을 쉽게 작성하도록 도와주는 python package로 [pre-commit](<https://pre-commit.com/#intro>)이 있다. pre-commit은 `.pre-commit-config.yaml`파일의 내용에 따라 hooks를 생성해준다. 홈페이지에서 제공하는 `.pre-commit-config.yaml`의 예시는 다음과 같다.

```bash
# .pre-commit-config.yaml
# repo: the repository url to git clone from
# rev: the revision or tag to clone at.
# hooks: A list of hook mappings.
#   - id: the id of the hook
#   - name: the name of the hook - shown during hook execution.
#   - entry: the entry point - the executable to run. 
#   - args: (optional: default []) list of additional parameters to pass to the hook.
#   - language: the language of the hook - tells pre-commit how to install the hook.
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks 
  rev: v2.3.0
  hooks:
    - id: trailing-whitespace
      name: Trim Trailing Whitespace
      description: This hook trims trailing whitespace.
      entry: trailing-whitespace-fixer
      language: python
      types: [text]
```

위의 Makefile에서 정의한 내용을 pre-commit hook으로 적용하기 위해서는 다음과 같이 `.pre-commit-config.yaml`을 작성하면 된다.

```
repos:
-   repo: local
    hooks:
    - id: format
      name: format
      entry: make format
      language: system
      types: [python]
    - id: lint
      name: lint
      entry: make lint
      language: system
      types: [python]
```

`.pre-commit-config.yaml`의 내용대로 hook을 생성하는 것은 다음 명령어로 가능하다.

```bash
pre-commit install
```

참고로 위의 명령어로 생성한 git pre-commit은 `.git/hooks/pre-commit` 파일에서 확인할 수 있다.
