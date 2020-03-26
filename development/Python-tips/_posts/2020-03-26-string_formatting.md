---
layout: post
title: 파이썬에서 String을 다루는 3가지 방법
category_num: 1
subtitle: 파이썬의 문자열 포맷팅 - % operator, format(), f'string'
---

# 파이썬에서 String을 다루는 3가지 방법

- update data : 2020.03.26
- 한 줄 요약 : 기호에 맞게 `format()` 또는 `f'string'` 방식을 사용하자.

## Methods

파이썬에서 문자열 포맷팅 방식은 크게 3가지로

- `% 연산자`를 사용하는 방법
- `.format()`를 사용하는 방법
- `f'string'`을 사용하는 사용하는 방법

등이 있다. `% 연산자`를 사용하는 방법은 C의 문자열 포맷팅 방식과 매우 유사한데, 다소 복잡하고 직관성이 떨어지기 때문에 많이 사용하지 않는다. `format()`과 `f'string`은 두 가지 방법 모두 개발자의 기호에 따라 자주 사용되고 있다.

### 1. % Operator

```python
>>> ops = "python formating no.%d" % 1
>>> print(ops)
python formating no.1
```

자주 사용되지 않기 때문에 간단하게 넘어가면, `%d`와 같이 문자열 속 변수를 대입하고자 하는 위치에 `%` 오퍼레이터를 사용하고, 문자열 뒤에 넣고자 하는 변수를 `% 변수`와 같이 호출하는 방식이다. 이때 변수의 자료형에 따라 operator % 뒤의 문자가 달라지는데, 정수의 경우 `d`, 문자열의 경우 `s`, 실수의 경우 `f`를 사용한다.

### 2. format()

#### 기본적인 사용 방식

`format()` method는 기본적으로 아래와 같이 사용한다. 문자열 내에 변수를 대입하고자 하는 위치를 중괄호 `{}`로 표현하고, `.format(*args)`에 parameter로 중괄호 순서에 맞춰 입력하는 방식이다.

```python
>>> ops = "python formating no.{} {}".format(2, "format()")
>>> print(ops)
python formating no.2 format()
```

아래와 같이 중괄호 내에 문자열을 입력하면 `format(**kwargs)` 방식으로 아래와 같이 사용할 수 있다.

```python
>>> ops = "python formating no.{number} {name}".format(number=2, name="format()")
>>> print(ops)
python formating no.2 format()
```

#### 실수(float)와 format method

`format()` method를 통해 소수점 이하의 크기가 매우 큰 실수(float)를 깔끔하게 표현할 수 있다. 머신러닝에서 accuracy 등과 같은 metric을 표현할 때 사용하면 좋다.

```python
>>> accuracy = 23.392324353534034232
>>> loss = 3231.23232424535

>>> print("accuracy : {:0.3f} loss : {:0.3f}".format(accuracy, loss))
accuracy : 23.392 loss : 3231.232
```

중괄호를 특정하기 위해 `:` 앞에 formating 함수의 parameter 순서를 의미하는 정수 또는 문자열을 넣을 수도 있다.

```python
>>> print("acc : {0:0.3f} loss : {1:0.3f}".format(accuracy, loss))
acc : 23.392 loss : 3231.232

>>> print("acc : {acc:0.3f} loss : {loss:0.3f}".format(acc=accuracy,loss=loss))
acc : 23.392 loss : 3231.232
```

### 3. f'string'

`f'string'` 방식은 format() method를 사용하는 방식보다 간결하다는 장점이 있다.

```python
>>> f_string_num=3

>>> ops = f"python formating no.{f_string_num}"
>>> print(ops)
python formating no.3
```

`f` 키워드로 formatting이 가능한 문자열이 되며, 포맷팅의 대상 변수는 중괄호 내에 곧바로 입력하는 방식이다.

```python
>>> print(f"accuracy : {accuracy:0.03f}")
accuracy : 23.392
```

과 같이 실수를 다루기 위해 format() method에서 사용한 방법도 동일하게 적용된다.
