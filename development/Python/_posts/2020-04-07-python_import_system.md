---
layout: post
title: Python import system
category_num: 2
---

# Python import system 이해하기

- update date : 2020.04.07

## Summary

- 절대 경로는 `sys.path`에 포함된 경로를 탐색하고, 상대 경로는 `__name__`에 정의된 위치를 기준으로 탐색한다
- 실행 파일의 `__name__`은 항상 **"__main__"**이다.
  - 따라서 실행 파일에서는 상대 경로를 사용하면 안 된다.
- `__name__` 에서 상위 디렉토리가 정의되지 않은 경우 해당 파일을 최상단으로 본다.
  - 따라서 실행파일과 동일한 디렉토리에서 모듈 간 상대 경로로 import를 하면 찾을 수 없다.
- 실행 파일의 디렉토리는 자동으로 `sys.path`에 포함된다.
  - 따라서 외부에서 접근할 때에는 절대 경로를 이용해 패키지 내부 모듈 간 import 하려면 상위 디렉토리까지 포함해야한다.
- 상위 디렉토리에서 접근하는 경우 `__name__`에 상위 디렉토리가 포함되는 꼴이 된다.
  - 따라서 이 경우 상대 경로를 이용해 패키지 내부 모듈 간 import 가 가능해진다.

## Python PEP

`PEP 328`과 `PEP 338`에는 다음과 같은 내용들이 정의되어 있다.

#### PEP 328

- [link](<https://www.python.org/dev/peps/pep-0328/>)

##### Absolute import

```
Rationale for Absolute Imports

refers to a top-level module or to another module inside the package. As Python's library expands, more and more existing package internal modules suddenly shadow standard library modules by accident. It's a particularly difficult problem inside packages because there's no way to specify which module is meant. To resolve the ambiguity, it is proposed that foo will always be a module or package reachable from sys.path. This is called an absolute import.
```

- 절대 경로는 `sys.path`로 접근 가능한 패키지 또는 모듈을 기준으로 탐색한다.

##### Relative import

```
Relative Imports and __name__

Relative imports use a module's __name__ attribute to determine that module's position in the package hierarchy. If the module's name does not contain any package information (e.g. it is set to '__main__') then relative imports are resolved as if the module were a top level module, regardless of where the module is actually located on the file system.
```

- 상대경로는 `__name__`을 기준으로 결정된다.
- `__name__`이 **"__main__"**인 경우를 비롯하여 패키지의 정보를 담고 있지 않으면 해당 모듈을 파일 시스템의 최상단에 위치한 것으로 간주한다. 

#### PEP 338

- [link](<https://www.python.org/dev/peps/pep-0338/>)

```
This is due to the fact that relative imports rely on __name__ to determine the current module's position in the package hierarchy. In a main module, the value of __name__ is always '__main__', so explicit relative imports will always fail (as they only work for a module inside a package)
```

- 실행파일(main module)의 `__name__`의 값은 항상 `"__main__"`이다. 따라서 상대 경로를 이용하면 항상 실패한다.

## Test Cases

PEP에 정의된 내용들을 확인하기 위해 간단히 실험을 진행해보았다.

### Exeample Directory Structure

test에 사용한 디렉토리 구조는 다음과 같다.

```
module_system
  |-main.py
  |-pkg1
    |- __init__.py
    |- module1.py
    |- module11.py
    |- module111.py
```

### 1. Execute in package

`module1.py`에서 `module11.py`의 함수를 import 하는 경우와 같이 동일한 디렉토리에서 import 하는 방법에 대해 실험을 진행했다.

#### 1.1. Absolute path + Execute in package => (O)

```python
# pkg1/module1.py

from module11 import return_module11_name
from module11 import get_module111_name


def return_module1_name():
    return str(__name__)

def get_module11_name():
    return return_module11_name()

print(return_module1_name())
print(get_module11_name())
print(get_module111_name())

```

```python
# pkg1/module11.py

from module111 import return_module111_name

def return_module11_name():
    return str(__name__)

def get_module111_name():
    return return_module111_name()

```

```python
# pkg1/module111.py

def return_module111_name():
    return str(__name__)
```

복잡해 보일 수 있으나, 각 module에서 __name__ 값을 반환하여 module1.py를 실행하면 모두 출력되도록 하는 코드이다. 여기서 module1.py 를 실행하면 문제 없이 정상적으로 아래와같이 출력되는 것을 확인할 수 있다.

```
$ python pkg1/module1.py
__main__
module11
module111
```

---

#### 1.2. Relative path + Execute in package => (X)

그런데 패키지 내에서 실행할 때 상대경로를 이용하면 문제가 된다. 아래와 같이 실행하고자 하는 파일 `module1.py`를 다음과 같이 상대경로로 바꾸면 에러가 발생한다.

```python
# pkg1/module1.py

from .module11 import return_module11_name
from .module11 import get_module111_name


def return_module1_name():
    return str(__name__)

def get_module11_name():
    return return_module11_name()

print(return_module1_name())
print(get_module11_name())
print(get_module111_name())

```

에러 내용은 아래와 같다.

```
Traceback (most recent call last):
  File "pkg1/module1.py", line 1, in <module>
    from .module11 import return_module11_name
ModuleNotFoundError: No module named '__main__.module11'; '__main__' is not a package
```

---

`module1.py`는 절대 경로로 하고, import 대상이 되는 `module11.py`에서만 상대경로로 접근하더라도 문제가 발생한다.

```python
# pkg1/module11.py

from .module111 import return_module111_name

def return_module11_name():
    return str(__name__)

def get_module111_name():
    return return_module111_name()
```

이때의 에러 내용은 다음과 같다.

```
Traceback (most recent call last):
  File "pkg1/module1.py", line 1, in <module>
    from module11 import return_module11_name
  File "/Users/enfow/projects/python_test/module_system/pkg1/module11.py", line 1, in <module>
    from .module111 import return_module111_name
ImportError: attempted relative import with no known parent package
```

---

두 가지 경우의 에러 내용이 다른 것을 확인할 수 있다.

- 첫 번째의 경우 `ModuleNotFoundError` 로, 이는 `__main__.module11` 라는 파일이 존재하지 않기 때문에 생기는 문제이다.
- 두 번째는 `ImportError` 이다. 이는 에러 코드에서도 명시되어 있듯이 `pkg1/` 디렉토리를 찾을 수 없어 생기는 문제이다.

**PEP 338**에서는 실행되는 모듈의 `__name__`은 **"__main__"**이 된다고 정의하고 있으며, 이러한 이유로 첫 번째 에러가 발생하는 것이다. 두 번째 에러는 **PEP 328**의 내용과 관련된 것으로, `__name__`에서 상위 디렉토리인 `pkg1/`를 찾을 수 없기 때문에 최상단으로 간주되고, 이로 인해 동일한 디렉토리에 위치한 다른 module을 찾을 수 없는 경우이다.

---

### 2. Execute out of package

`main.py`와 같이 package 외부에 있는 파일에서 import 한다면 어떻게 될지도 실험을 통해 확인해보았다.

#### 2.1. Absolute path + Execute out of package => (X, O)

```python
# pkg1/module1.py

from module11 import return_module11_name
from module11 import get_module111_name


def return_module1_name():
    return str(__name__)

def get_module11_name():
    return return_module11_name()

print(return_module1_name())
print(get_module11_name())
print(get_module111_name())
```

위와 같이 첫 번째 예시와 동일하게 `module1.py` 가 절대경로로 정의되어 있다고 할 때,

```python
# main.py
from pkg1.module1 import return_module1_name
from pkg1.module1 import get_module11_name

print(return_module1_name())
print(get_module11_name)
```

main.py 를 실행하게 되면 다음과 같은 에러가 발생한다.

```
Traceback (most recent call last):
  File "main.py", line 3, in <module>
    from pkg1.module1 import return_module1_name
  File "/Users/enfow/projects/python_test/module_system/pkg1/module1.py", line 1, in <module>
    from module11 import return_module11_name
ModuleNotFoundError: No module named 'module11'
```

이는 절대경로를 이용할 때 모듈을 찾는 방식과 관련된다. 절대경로의 경우 `sys.path`에 포함된 경로에서만 패키지 또는 모듈을 찾게 된다. 현재 main.py 가 실행되는 위치는 `module_system` 디렉토리이고, module11.py는 `module_system/pkg1/module11.py`에 저장되어 있으므로 `module_system/`에서는 찾을 수 없다.

이러한 문제를 해결하기 위한 방법은 크게 세 가지로 다음과 같다.

- 패키지 내에서는 상대 경로를 사용하는 방법
- 패키지 내에서도 항상 package 이름을 포함해 절대 경로를 사용하는 방법
- sys.path에 포함되도록 강제하는 방법

---

#### 2.2. Relative path + Execute out of package => (O)

아래와 같이 `pkg1/` 내의 모든 모듈에서 절대 경로를 상대 경로로 바꾸면 에러 없이 잘 동작한다.

```python
# pkg1/module1.py

from .module11 import return_module11_name
from .module11 import get_module111_name


def return_module1_name():
    return str(__name__)

def get_module11_name():
    return return_module11_name()

print(return_module1_name())
print(get_module11_name())
print(get_module111_name())

```

```python
# pkg1/module11.py

from .module111 import return_module111_name

def return_module11_name():
    return str(__name__)

def get_module111_name():
    return return_module111_name()

```

이때 `main.py` 를 실행하면,

```python
# main.py
from pkg1.module1 import return_module1_name
from pkg1.module1 import get_module11_name

print(return_module1_name())
print(get_module11_name)
```

다음과 같이 출력된다.

```
$ python main.py
pkg1.module1
pkg1.module11
pkg1.module111
pkg1.module1
pkg1.module11
```

여기서 처음 세 줄은 `pkg1/module1.py`의 내용이 출력된 것이고, 마지막 두 줄이 `main.py` 에서 출력된 것이다.
