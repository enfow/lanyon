---
layout: post
title: Linting & Formatting
category_num : 6
keyword: '[Tools]'
---

# Linting & Formatting

- update date: 2020.11.08
- Environement Setttings: MacOS 10.15.6 \| Python 3.7.4

## Static & Dynamic Analysis

내가 작성한 소스코드에 문제가 없는지 알아보는 방법은 크게 **정적 분석(Static Analysis)**과 **동적 분석(Dynamic Analysis)** 두 가지로 나누어진다. 두 가지를 구분하는 기준은 소스코드의 실행 여부라고 할 수 있는데, 정적 분석은 코드를 런타임에 올리지 않은 채 텍스트로 보고 코드를 분석하는 것을, 동적 분석은 런타임에 올려 실제로 확인한 실행 결과를 바탕으로 코드를 분석하는 것을 가리킨다. 정적 분석의 대표적인 예로는 **Linting**과 **Formatting** 등이 있고, 동적 분석의 대표적인 예로는 **Unit Test**가 있다.

정적 분석과 동적 분석은 상호 보완적인 관계로 그 특성이 다르기 때문에 서로 잡아낼 수 있는 문제점 또한 다르다. 따라서 완벽성을 추구하기 위해서는 두 가지를 모두 사용하는 것이 좋다. 다만 코드가 런타임에서 어떻게 동작하는지 확인해야 하는 동적 분석은 정적 분석에 비해 무거울 수 밖에 없다. 따라서 큰 프로젝트를 수행할 때에는 하위 브랜치에서 작업할 때에는 정적 분석만 실시하고, master, develop과 같은 주요 브랜치에 merge 할 때에만 동적 분석을 실시하는 등 상황을 고려하여 유연하게 적용할 필요가 있다.

본 포스팅에서는 정적 분석 방법인 Linting과 Formatting이 무엇인지 알아보고 Python에서 사용할 수 있는 도구들을 확인해보고자 한다.

---

## Linter & Formatter

#### Linter

위키에서는 [Linter](<https://en.wikipedia.org/wiki/Lint_(software)>)를 다음과 같이 정의하고 있다.

- "lint, or a linter, is a **static code analysis tool** used to flag **programming errors, bugs, stylistic errors, and suspicious constructs**."

정리하면 Linter는 소스 코드에서 프로그래밍 에러나 버그, 구조적인 문제점 등을 잡아내는 정적 분석기를 말한다. Python에서 많이 사용되는 Linter로는 **pylint**와 **flake8**이 있다.

#### Formatter

코드가 정해진 목적에 따라 동작하지 않는 것을 가리키는 Code Error(또는 Bug)와 달리 Code Format은 옳고 그름의 문제가 아니고 개발자 개인의 선호의 문제이지만 여러 사람이 함께 작업할 때에는 가독성 등의 이유로 웬만하면 동일한 Code Format을 정하고 그것에 맞춰 작업하게 된다. 말이 쉽지 여러 사람이 함께 Format을 정하고 자신의 선호와 다른 Format에 맞춰 코드를 작성하는 것은 결코 쉽지 않다. Code Formatter는 이러한 문제를 해결하기 위한 도구로서 개인이 작성한 코들르 일정한 Code Format에 맞춰준다. Python의 대표적인 Code Formatter로는 **Black**이 있다.

## pylint

Python에서 가장 대표적인 linter라고 할 수 있는 [pylint](<https://github.com/PyCQA/pylint>)는 다음과 같이 pip install 로 쉽게 설치가 가능하다.

```bash
$ pip install pylint
```

사용하는 것도 간단하다.

```bash
$ pylint [file_name || dir_name]
```

예를 들어  다음과 같은```main.py```에 pylint를 실행한 결과는 다음과 같다.

```python
# main.py

print("It is main")

def sum(a, b):
    return np.sum(10, 20)
```

```bash
# pylint
************* Module main
main.py:4:0: C0304: Final newline missing (missing-final-newline)
main.py:3:0: W0622: Redefining built-in 'sum' (redefined-builtin)
main.py:1:0: C0114: Missing module docstring (missing-module-docstring)
main.py:3:0: C0103: Argument name "a" doesn't conform to snake_case naming style (invalid-name)
main.py:3:0: C0103: Argument name "b" doesn't conform to snake_case naming style (invalid-name)
main.py:3:0: C0116: Missing function or method docstring (missing-function-docstring)
main.py:4:11: E0602: Undefined variable 'np' (undefined-variable)
main.py:3:8: W0613: Unused argument 'a' (unused-argument)
main.py:3:11: W0613: Unused argument 'b' (unused-argument)

----------------------------------------------------------------------
Your code has been rated at -33.33/10 (previous run: -6.67/10, -26.67)
```

디렉토리를 단위로도 실행할 수 있다. 이 경우 디렉토리에 포함된 모든 파일들에 대한 분석 결과를 반환한다. 해당 디렉토리의 하위 디렉토리까지도 모두 검사하며, Python 파일이 아닌 경우에는 무시한다.

```
dir
 |-inner_dir
  |-file3.py
 |-file1.py
 |-file2.py
 |-it_is_c.c
```

```bash
# result of "$ pylint dir"
************* Module dir.file2
dir/file2.py:1:0: C0304: Final newline missing (missing-final-newline)
dir/file2.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module dir.file1
dir/file1.py:1:0: C0304: Final newline missing (missing-final-newline)
dir/file1.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module dir.inner_dir.inner_file1
dir/inner_dir/inner_file1.py:1:0: C0304: Final newline missing (missing-final-newline)
dir/inner_dir/inner_file1.py:1:0: C0114: Missing module docstring (missing-module-docstring)
```

#### pylint with vscode

vscode를 사용하면 `Python` extention을 통해 IDE에서 pylint를 적용한 결과를 곧바로 확인할 수 있다.

<img src="{{site.image_url}}/development/pylint_python_extention.png" style="width:40em; display: block; margin-top: 2em; margin-bottom: 2em">

설치 후 `command + shift + p`로 vscode command prompter에 진입하여 `Python: Select Linter`를 선택한다.

<img src="{{site.image_url}}/development/pylint_pylint_setting1.png" style="width:35em; display: block; margin-top: 2em; margin-bottom: 2em">

여기서 `pylint`를 선택한다.

<img src="{{site.image_url}}/development/pylint_pylint_setting2.png" style="width:35em; display: block; margin-top: 2em; margin-bottom: 2em">

이렇게하면 `.vscode/settings.json` 파일이 생성된다. 여기서 vscode에 적용되는 pylint 시스템을 관리할 수 있다. 처음 생성된 경우 pylintEnabled와 enabled 두 가지 옵션만 입력되어 있는데, pylint를 제대로 적용하기 위해서는 사용하는 python의 PythonPath를 입력해주어야 한다. 입력 방법은 크게 두 가지인데, 직접 settings.json을 아래와 같이 수정하는 방법이 있고, vscode 좌측 하단에서 사용하고자 하는 Python Interpreter를 선택하는 방법이 있다.

```json
/*.vscode/settings.json*/
{
    "python.pythonPath": "/Users/***/[python path]",
    "python.linting.pylintEnabled": true,
    "python.linting.enabled": true
}
```

관련 내용은 [vscode docs](<https://code.visualstudio.com/docs/python/linting>)에 정리되어 있다. 참고로 vscode에서 Pylint를 적용하게 되면 다음과 같이 빨간 줄로 문제가 있는 부분을 표시해준다.

<img src="{{site.image_url}}/development/pylint_vscode_error.png" style="width:35em; display: block; margin-top: 2em; margin-bottom: 1em">

---

## flake8

또다른 python linter인 [flake8](<https://pypi.org/project/flake8/>) 또한 기본적인 사용 방식은 pip로 설치가 가능하고 파일 또는 디렉토리 단위로 검사할 수 있다는 점에서 pylinter와 거의 동일하다.

```bash
# install flake8
$ pip install flake8
```

```bash
# execute flake8
$ flake8 main.py
```

위의 pylint에서 사용한 예시와 동일한 main.py 파일에 대한 flake8의 결과는 다음과 같다.

```bash
main.py:3:1: E302 expected 2 blank lines, found 1
main.py:4:12: F821 undefined name 'np'
main.py:4:26: W292 no newline at end of file
```

### flake8 plugins

flake8의 장점 중 하나는 다양한 plug-ins가 있어 검사 방법 등을 필요에 따라 변경하는 것이 가능하다는 것이다. 대표적인 flake8 plugin은 다음 레포에 잘 정리되어 있다: [Awesome Flake8 Extensions](<https://github.com/DmytroLitvinov/awesome-flake8-extensions>)

- flake8-builtins
- pep8-naming
- flake8-docstrings
- flake8-comprehensions

### flake8-builtins

[flake8-builtins](<flake8-builtins>)는 list, dict 등과 같이 Python에 내장되어 있는 변수 또는 파라미터를 새로운 변수 명으로 사용하는지 확인해주는 plug-in이다.

```bash
# install flake8-builtins
$ pip install flake8-builtins
```

로 설치할 수 있으며 예시는 다음과 같다.

```python
# main.py
def sum(list):
    return np.sum(list)
```

```bash
# result of "$ flake8 main.py"
...
main.py:3:1: A001 variable "sum" is shadowing a python builtin
main.py:3:9: A002 argument "list" is shadowing a python builtin
...
```

sum과 list로 미리 정의된 Python 함수가 있다는 것을 알려주고 있다.

### pep8-naming

[pep8-naming](<https://github.com/PyCQA/pep8-naming>)는 말 그대로 pep8 규약에 맞게 naming이 이루어졌는지 확인해주는 plug-in이다. 아래 표를 보면 알 수 있듯 class, function 등의 이름을 지을 때 대소문자를 정확하게 적용하였는지에 관한 것들이다.

```bash
# install pep8-naming
pip install pep8-naming
```

|Code|Explanation|
|---|---|
|N801|class names should use CapWords convention|
|N802|function name should be lowercase|
|N803|argument name should be lowercase|
|N804|first argument of a classmethod should be named 'cls'|
|N805|first argument of a method should be named 'self'|
|N806|variable in function should be lowercase|
|N807|function name should not start and end with '__'|
|N811|constant imported as non constant|
|N812|lowercase imported as non lowercase|
|N813|camelcase imported as lowercase|
|N814|camelcase imported as constant|
|N815|mixedCase variable in class scope|
|N816|mixedCase variable in global scope|
|N817|camelcase imported as acronym|

### flake8-docstrings

[flake8-docstrings](<https://gitlab.com/pycqa/flake8-docstrings>)은 Documentation을 정해진 양식에 맞게 작성하도록 강제하는 plug-in이다. 여기서 정해진 양식이란 [pydocstyle](<http://www.pydocstyle.org/en/latest/usage.html>)을 말하며, Error Code 역시 [pydocstyle/error codes](<http://www.pydocstyle.org/en/latest/error_codes.html>)에 정리되어 있다.

```bash
# install flake8-docstrings
$ pip install flake8-docstrings
```

모든 plug-in이 그렇지만 사전에 정의된 규약이 익숙치 않으면 지키는 것에 어려움을 겪기 마련이다. 익숙치 않으면 다음과 같이 반복적으로 Docstring Error를 만나게 된다.

#### Trial 1

```python
list_example = [i for i in range(10)]

def sum_operation(list):
    return list[0] + list[1]

a = sum_operation(list_example)

```

```bash
main.py:1:1: D100 Missing docstring in public module
main.py:3:1: D103 Missing docstring in public function
```

#### Trial 2

list_example과 sum_operation에 맞는 docstring이 없다는 것이므로 추가해준다.

```python
"""list_example"""
list_example = [i for i in range(10)]

def sum_operation(list):
    """sum_operation"""
    return list[0] + list[1]

a = sum_operation(list_example)
```

```bash
main.py:1:1: D400 First line should end with a period
main.py:5:1: D400 First line should end with a period
```

#### Trial 3

이유는 모르겠지만 docstring의 첫 번째 라인은 항상 온점으로 끝나야 한다고 한다. 이것까지 수정해주고 나면 더 이상 docstring error가 발생하지 않는다.

```python
"""list_example."""
list_example = [i for i in range(10)]

def sum_operation(list):
    """sum_operation."""
    return list[0] + list[1]

a = sum_operation(list_example)
```

### flake8-comprehensions

[flake8-comprehensions](<https://github.com/adamchainz/flake8-comprehensions>)는 list, set, dict 등을 선언할 때 보다 이해하기 쉬운 방법으로 작성하도록 하는 plug-in 이다. plug-in 레포에서 보여주고 있는 예시는 다음과 같다.

- Rewrite `dict()` as `{}`
- Rewrite `dict(a=1, b=2)` as `{"a": 1, "b": 2}`
- Rewrite `list()` as `[]`
- Rewrite `list(f(x) for x in foo)` as `[f(x) for x in foo]`
- Rewrite `sum([x ** 2 for x in range(10)])` as `sum(x ** 2 for x in range(10))`

---

## Black

Python Code Formatter인 [Black](<https://pypi.org/project/black/>)에 대한 소개 문구는 다음과 같다.

- Black is the uncompromising **Python code formatter**. By using it, you agree to cede control over minutiae of hand-formatting. In return, **Black gives you speed, determinism, and freedom from pycodestyle nagging** about formatting. You will save time and mental energy for more important matters.

빠르게 pycodestyle의 잔소리로부터 해방시켜 준다고 한다. 그도 그럴것이 black을 사용하면 매우 빠르고 쉽게 임의로 작성한 코드를 pycodestyle에 맞춰준다. 우선 black의 설치 방법은 다음과 같다.

```bash
# install black
$ pip install black
```

실행 방법도 매우 간단하다.

```bash
# execute black
$ black [file_name or dir_name]

# check option
$ black [file_name or dir_name] --check
```

예를 들어 다음과 같이 main.py 파일이 주어져 있다면

```python
# main.py
list_example = [i for i in range(10)]

def sum_operation(list):
    return list[0] + list[0]

a = sum_operation(list_example)

s
```

다음과 같이 변경해준다.

```python
# main.py
list_example = [i for i in range(10)]


def sum_operation(list):
    return list[0] + list[0]


a = sum_operation(list_example)

s
```

이때 실행 결과 메시지는 다음과 같다.

```bash
reformatted main.py
All done! ✨ 🍰 ✨
1 file reformatted.
```

위의 실행 결과를 비교해 보면 변수와 함수의 선언 사이에 띄어쓰기를 추가했음을 알 수 있다. 그런데 위의 Formatting된 코드를 실행해보면 마지막 줄의 's' 때문에 다음과 같은 에러 메시지를 만나게 된다.

```bash
Traceback (most recent call last):
  File "main.py", line 10, in <module>
    s
NameError: name 's' is not defined
```

즉 Code Formatter는 Code의 양식을 맞춰주는 것일 뿐 오류를 자동으로 수정해주거나 하지는 않는다.
