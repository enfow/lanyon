---
layout: post
title: "Testings for Machine Learning: Unit Test"
category_num : 2
---

# Testings for Machine Learning: Unit Test

- 머신러닝 리서치 엔지니어로 일하면서 느낀 Unit Test의 필요성 및 PyTest를 활용한 Test Code 작성 방법에 대해 정리해 보았습니다.
- [PyTest](<https://docs.pytest.org/en/stable/>)
- Update at: 2021.01.23

## Why Unit Test?

Unit Test란 어떤 한 프로그램의 소스코드를 구성하는 개별 코드 조각(Unit)들이 원하는 대로 정확히 동작하는지 검증하는 Software Testing 방법을 말한다. 단순하게 생각하면 프로그램을 구성하는 함수들이 특정 입력에 대해 기대되는 값을 출력하는지 일일이 확인하는 작업이다. 따라서 Unit Test에서는 특정 함수가 프로그램에서 어떤 위치에 있는지에 대해서는 전혀 관심이 없다. 전체 프로그램이 무엇을 위한 것이든 간에 Unit Test의 관점에서 보면 조각의 모음에 불과하다고 할 수 있다.

### Unit Test for Machine Learning

머신러닝에서는 구현한 모델에 대한 성능 검증이 필수적이고, 기존의 방법론과 비교하여 약간이라도 높은 성능을 보이는 것이 중요하게 여겨진다. 그런데 모델을 구현하고 실험 환경을 구축하는 과정에서 어느 정도의 휴먼 에러는 발생할 수밖에 없다. 대표적으로 다음과 같은 경우들을 생각해 볼 수 있다.

- Test Set에 Training Set이 일부 섞여 들어가는 경우
- Trianing Set에 포함되어선 안 되는 레이블의 데이터가 포함되는 경우
- MinMax Scaling에서 Max 값이 잘못 설정되어 1 이상의 값이 나오는 경우

위의 예시들이 가지는 공통점 중 하나는 모델의 학습 과정 상에서는 어떠한 에러도 발생시키지 않는다는 것이다. 이 중 일부는 모델의 성능에 직접적인 영향을 미치기 때문에 특히 조심해야 함에도 불구하고 학습 자체는 정상적으로 이뤄지는 것처럼 보여 모르고 넘어가는 경우가 많다. Unit Test를 잘 수행하면 위에서 언급한 문제들 뿐만 아니라 실험 과정에서 발생할 수 있는 여러 오류들을 잡아낼 수 있다.

### Regression Test

그런데 머신러닝에서는 그 특성상 하드코딩으로 Unit Test를 수행하기 어려운 경우가 있다. 예를 들어 평균이 0이고 분산이 1인 정규 분포에서 랜덤한 샘플을 반환하는 함수 `sample_gaussian()`이 있어 이를 검증한다고 할 때 다음과 같이 Test Code를 작성하는 것은 좋은 선택이 되지 못한다. 왜냐하면 연속 공간에서 특정 숫자가 뽑힐 확률은 0이기 때문이다.

```python
def test_gaussian_sampling(self):
    sample = sample_gaussian()
    assert sample == 0.0
```

이러한 경우에는 Regression Test를 사용하여 코드를 검증하게 된다. 참고로 본 포스팅에서는 이러한 케이스들에 대해서는 다루지 않을 예정이다.

## PyTest

**PyTest**는 이름에서도 알 수 있듯이 Python에서 Unit Test를 위해 자주 사용되는 framework이다. 설치 방법을 비롯한 기본적인 사용 방법은 [PyTest 홈페이지](<https://docs.pytest.org/en/stable/getting-started.html>)에 잘 나와있다.

### Conventions for Test Codes

PyTest는 다음과 같이 간단한 명령어로 실행할 수 있다.

```bash
$ pytest
```

그런데 위의 명령어를 실행했을 때 유효한 Test로 인식하기 위해서는 다음과 같은 Directory Convention을 따라야 한다.

- Test File들은 모두 testpaths 내에 위치해야 한다. 일반적으로 testpaths는 `tests/`로 한다.
- testpaths 내에서는 디렉토리를 생성하여 Test File들을 분류할 수 있다. Recursive하게 testpaths 내의 디렉토리를 탐색하기 때문이다.
- testpaths 내의 Test File의 이름은 모두 `test_*.py` 또는 `*_test.py` 꼴이어야 한다.

PyTest에서 제공하는 [디렉토리 구조의 예시](<https://docs.pytest.org/en/stable/example/pythoncollection.html>)는 다음과 같다.

```
tests/
|-- example
|   |-- test_example_01.py
|   '-- test_example_02.py
|-- foobar
|   |-- test_foobar_01.py
|   '-- test_foobar_02.py
'-- hello
    '-- world
        |-- test_world_01.py
        '-- test_world_02.py
```

### Writing Test Codes

각 Test File에는 Class의 Method 혹은 Function의 형태로 정의된 Test Case들이 정의되어 있다. Class의 Method로 정의하느냐, Function으로 정의하느냐 큰 차이는 없고 모두 하나의 Test Case를 공유한다는 점에서는 동일하다. Function으로 정의하는 것이 보다 간편하지만 Class로 정의하면 Test의 결과를 빠르게 이해하는 데에 도움이 되고, 설정 값 등을 공유할 수 있어 편리하다는 서로 다른 장점이 있다.

Test Case들에 대해서도 Convention을 따라야 PyTest가 Test Case로 인식할 수 있는데, 대표적으로는 다음과 같은 것들이 있다.

- Test File 내에 정의된 모든 Test Case들은 Class의 Method 혹은 Function 형태로 정의되어야 하며, 이때 Class와 Method, Function의 이름은 모두 `test_*` 꼴이어야 한다.
- Class의 이름은 `Test*.py`여야 한다.
- Class를 사용한다면 생성자 `__init__()` Method를 만들지 말아야 한다.

이해를 돕고자 TDD를 공부하며 개인적으로 작성한 [예시](<https://github.com/enfow/test-driven-dev-python/blob/main/tests/test_ch1.py>)를 추가해 보았다.

```python
# tests/test_ch1.py
class TestDollar:
    """Test Dollar"""

    def test_multiplication(self):
        """test code"""
        five = Dollar(5)
        five.times(2)
        assert 10 == five.amount
```

### Make Test Easier

Unit Test 뿐만 아니라 모든 Test는 그 과정이 매우 쉽고 부담이 적어야 한다고 생각한다. 그래야 보다 자주 Testing을 수행하고, 문제를 더욱 빠르게 찾아낼 수 있기 때문이다. pytest는 이러한 점에서 명령어가 매우 간단(`$ pytest`)하기 때문에 이러한 이상에 부합한다. 하지만 pytest 또한 option 값을 넣어야 한다면 다소 복잡해질 수 있다. 이때 사용하는 것이 `Makefile`이다.

```bash
# Makefile
utest:
	env PYTHONPATH=. pytest ./tests/ -s --verbose --ignore tests/test_example_01.py
```

위와 같이 설정해두면 아래 명령어로 `./tests` 디렉토리에서 Test Case들을 찾되 test_example_01.py는 무시하는 Unit Test를 수행할 수 있다. 추가적인 option 값들은 [PyTest 홈페이지](<https://docs.pytest.org/en/stable/reference.html#command-line-flags>)에서 확인 가능하다. 참고로 `env PYTHONPATH=.`는 때때로 PyTest가 디렉토리를 잘못 잡는 경우가 있어 방어적으로 추가한 것이라고 이해하면 된다.
