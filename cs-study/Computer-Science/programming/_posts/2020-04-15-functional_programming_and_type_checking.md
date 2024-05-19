---
layout: post
title: Functional Programming and Static Type Checking With Ocaml
category_num: 501
keyword: "[Programming]"
---

# Functional Programming and Static Type Checking With Ocaml

- update date : 2020.04.15

## 0. Introduction

**`Ocaml`**은 함수형 프로그래밍과 정적 타입 추론이라는 특성을 가지고 있다. 여기서 **함수형(Functional) 프로그래밍**은 프로그래밍 언어 패러다임의 일종으로, 주류 패러다임이라 할 수 있는 명령형(Imperative) 프로그래밍에 속하는 여러 언어들이 함수형 프로그래밍의 특성을 일부 포함하면서 유명해졌다. 이러한 점에서 함수형 프로그래밍은 미래 프로그래밍 패러다임으로 불리기도 한다. **정적(Static) 타입 추론**의 경우 컴파일러 단에서 타입 추론을 실시하는 것을 말한다. 이렇게 컴파일 과정에서 타입 추론을 하게되면 컴파일 이후 프로그램이 동작하는 과정에서 예상치 못한 에러로 인해 갑자기 종료되는 상황을 줄일 수 있어 프로그램의 안정성을 높여주는 특성으로 알려져 있다. 본 포스팅에서는 이러한 Ocaml의 두 가지 특성을 정리해 보았다.

---

## 1. Functional Programming Langauge

- `OCaml is a functional programming language`

[Ocaml 공식 홈페이지](https://ocaml.org/learn/description.html)에서는 ocaml을 함수형 프로그래밍 언어라고 정의한다. 이와 관련하여 위키에서는 [함수형 프로그래밍](https://en.wikipedia.org/wiki/Functional_programming)에 대해 다음과 같이 언급하고 있다.

```
In computer science, functional programming is a programming paradigm—a style of building the structure and elements of computer programs—that treats computation as the evaluation of mathematical functions and avoids changing-state and mutable data. It is a declarative programming paradigm in that programming is done with expressions or declarations instead of statements.
```

여기서 함수형 프로그래밍은 다음 두 가지 주요 특징을 가지는 것을 알 수 있다.

- 데이터의 가변성이 제한된다. 형 변환, 값 변환 등이 없다.
- statement를 사용하지 않고 declaration, expression 을 중심으로 동작한다.

이것의 의미는 명령형 프로그래밍과 비교해 보면 보다 쉽게 이해할 수 있다.

### Imperative Programming VS Functional Programming

```python
# python
d = 10
```

최근 많이 사용되는 언어이자 대표적인 명령형 언어인 **python**에서는 객체 `d`에 integer 10을 넣는다고 표현한다. 이는 프로그램 내에 `d`라는 data field(state)가 존재하고 해당 영역에 저장된 값을 10으로 바꿔주는 것을 의미한다고 할 수 있다. 그리고 이와 같이 state를 변경하는 것을 **statement**라고 한다.

```ocaml
# ocaml
let d = 10;;
```

반면 함수형 프로그래밍인 **ocaml**에서는 integer 10에 `d`라는 이름을 붙였다고 표현한다. 함수형 프로그래밍에서는 명령형 언어와 달리 어떠한 state도 바뀌지 않는다. 정확하게 말해 함수형 프로그래밍에서는 state가 존재하지 않고, 그저 integer 10을 부르는 다른 이름이 생겼다고 보아야 한다. 바꾸어야 할 state가 존재하지 않으므로 당연히 state를 바꾸는 역할을 하는 **statement**도 함수형 프로그래밍에는 존재하지 않는다. 오직 declaration과 expression 밖에 없다.

이와 관련하여 [wiki](https://en.wikipedia.org/wiki/Comparison_of_programming_paradigms)에서는 다음과 같이 명령형 프로그래밍과 함수형 프로그래밍을 다음과 같이 정의하고 있다.

- **Imperative Programming**: focuses on how to execute, defines control flow as statements that change a program state.
- **Functional Programming**: treats programs as evaluating mathematical functions and avoids state and mutable data

추가적으로 [Microsoft](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/linq/functional-programming-vs-imperative-programming)에서는 다음과 같이 두 가지 패러다임의 특성을 구별한다.

- **Imperative Programming**: How to perform tasks (algorithms) and how to track changes in state.
- **Functional Programming**: What information is desired and what transformations are required.

---

## 2. Static type checking

ocaml 홈페이지에서는 ocaml이 정적 타입 검사를 사용한다고 한다.

```
Although OCaml is statically type-checked, it does not require that the types of function parameters, local variables, etc. be explicitly declared, contrary to, say, C or Java. Much of the necessary type information is automatically inferred by the compiler.
```

정적 타입 검사와 동적 타입 검사의 차이는 타입 검사의 시점과 관련 있다. 정적 타입 검사는 컴파일 과정에서 실시되기 때문에 프로그램의 실행 과정에서 타입이 변경되지 않는다. 반면 동적 타입 검사의 경우 런타임 과정에서 타입 검사가 이뤄지며, 경우에 따라 데이터 타입이 변경되기도 한다.

### C language - static type checking

대표적인 정적 타입 검사 언어로 C 와 JAVA가 있다. 이들 언어에서는 사용자가 변수의 type을 직접 입력해줘야 한다.

```c
int main() {
    int a = 10;
    char b = "boy";
    return 0
}
```

### Ocaml - static type checking

반면 ocaml에서는 타입 선언이 명시적으로 이뤄지는 경우가 없다. 컴파일러에서 타입의 명시가 없더라도 적절한 타입을 추론하여 사용하기 때문이다.

```ocaml
# 10;;
- : int = 10
# 'a';;
- : char = 'a'
```

타입에 대한 명시가 없어도 10을 입력하면 `int` 라는 것을, 'a' 를 입력하면 `char` 라는 것을 알려준다.

### Python - dynamic type checking

물론 python에서도 타입 명시가 직접적으로 이뤄지지는 않는다.

```python
>>> 10
10
>>> type(10)
<class 'int'>
```

출력값만 보면 비슷해 보이지만 python은 동적 타입 검사를 실시한다는 점에서 ocaml의 타입 검사와 차이가 있다. python을 비롯한 동적 타입 검사 언어는 컴파일러에서 타입 검사가 이뤄지는 것이 아니라 런타임 과정에서 타입 추론이 이뤄진다. 따라서 프로그램이 실행되는 과정에서 타입이 변경될 수도 있는데, 이는 곧 코드 상의 문제로 인해 프로그래머가 의도하지 않은 데이터 타입으로 변경될 수도 있다는 것을 의미한다. 이러한 점에서 동작 타입 검사를 사용하는 언어들이 안전하지 않다고 말하는 프로그래머도 있다.
