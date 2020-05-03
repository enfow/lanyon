---
layout: post
title: Ocaml Basic grammar
category_num: 2
---

# Ocaml Basic grammar

- update date : 2020.04.11

## 0. Contents

1. Data Type
2. let : naming value
3. tuple and list
4. if … then … else
5. Pattern Matching
6. Function Declaration
7. Recursion not For/While

---

## 1. Data Type

ocaml은 다음 6가지의 기본 type 을 제공한다.

char | string | int | float | bool | unit

int의 경우 나머지 한 비트를 메모리 관리를 위해 사용하므로 64 비트 환경에서는 63비트, 32비트 환경에서는 31비트로 표현할 수 있는 수의 범위를 갖는다고 한다.

다른 언어에서 사용하는 명칭과 크게 다르지 않지만 unit 이라는 타입이 다소 생소하다. [OCaml for the Skeptical](<https://www2.lib.uchicago.edu/keith/ocaml-class/data.html>) 에 따르면 unit은 'no value' 를 의미한다고 한다. 함수에서 인자로 다른 함수를 받을 때 주로 사용한다. 대표적으로 함수를 인자로 받는 `List.iter` 의 경우 다음과 같이 정의된다.

```ocaml
# List.iter;;
- : ('a -> unit) -> 'a list -> unit = <fun>
```

기본 타입은 위 6가지이지만 built-in 타입으로 tupels, list, array 등을 지원한다.

### custom type

언어에서 지원하는 타입 외에도 사용자가 직접 새로운 데이터 타입을 만들 수 있다.

```ocaml
# type contry = Korea | England | Egypt;;
type contry = Korea | England | Egypt

# Korea;;
- : contry = Korea
```

`type` 키워드를 사용하여 country 라는 이름의 타입이 있고, Korea, England, Egypt 등이 이러한 타입을 가진다고 명시했다. 이후 Korea 를 입력하니 country 타입이라는 것을 알려준다.

중요한 것 중 하나로 데이터의 type을 선언할 때에는 대문자를 포함해야 한다는 것이다. 다음과 같이 선언하면 에러가 난다.

```ocaml
# type contry = korea | england | egypt;;
Error: Syntax error
```

---

## 2. let : naming value

let keyword는 다음과 같이 어떤 값의 이름을 정해줄 때 사용한다.

```ocaml
# let a = true;;
val a : bool = true

# let b = 'c';;
val b : char = 'c'

# let c = "string";;
val c : string = "string"

# let d = 10;;
val d : int = 10

# let e = 10.;;
val e : float = 10.
```

함수형 프로그래밍에 속하는 ocaml에는 state가 존재하지 않으므로, 값을 넣어주거나, 변경하는 것이 아닌 이름을 붙여준다는 표현이 보다 적절하다. 타입 추론은 자동적으로 이뤄지지만 다음과 같이 타입을 지정할 수도 있다.

```ocaml
# let d : int = 10;;
val d : int = 10
```

적절하지 않은 타입을 지정하면 당연히 동작하지 않는다.

```ocaml
# let d : float = 10;;
Error: This expression has type int
       but an expression was expected of type floa
```

### let ... in ...

let keyword 뒤에 in keyword를 덧붙이는 경우가 있다. 보다 정확하게 말하면 위의 방법이 in keyword를 생략해서 사용한 것으로 보아야 할 것이다.

```ocaml
let name = expr1 in expr2
```

이렇게 in keyword를 사용하게 되면 let keyword로 정의된 name 은 in keyword에 속하는 expression에서만 사용할 수 있다. scope를 제한하는 것이라고 이해하면 쉽다.

---

## 3. tuple and list

ocaml에도 tuple과 list와 같은 Data type이 존재한다.

### tuple

tuple은 다음과 같이 선언한다.

```ocaml
# let tp : int * int = (10, 10);;
val tp : int * int = (10, 10)
```

타입 추론이 자동적으로 이뤄지므로 다음도 가능하다.

```ocaml
# let tp = (10, 10);;
val tp : int * int = (10, 10)
```

서로 다른 타입으로 구성하는 것도 가능하다.

```ocaml
# let tp = (10, 10, 10.);;
val tp : int * int * float = (10, 10, 10.)
```

### list

리스트는 다음과 같이 선언할 수 있다.

```ocaml
# let li = [10;20;30];;
val li : int list = [10; 20; 30]
```

리스트를 타입과 함께 추론하려면 다음과 같이 `<datatype> list` 를 사용한다.

```ocaml
# let li : int list = [10;20;30];;
val li : int list = [10; 20; 30]
```

리스트의 경우 서로 다른 타입으로 구성할 수 없고, 모든 구성요소의 타입이 동일해야 한다.

```ocaml
# let li = [10;20;30.];;
Error: This expression has type float
       but an expression was expected of type int
```

---

## 4. if ... then ... else

ocaml에도 if else 가 있다. 정확하게 `if then else`이다.

```ocaml
# let tf = true;;
val tf : bool = true

# let ie = if tf then 1 else 0;;
val ie : int = 1
```

`tf`가 `true` 이므로, `then` 키워드의 값인 integer 1이 `ie`가 되었다. 반대로 `tf`가 `false`였다면 `else` 키워드의 값인 integer  0으로 되었을 것이다.

---

## 5. Pattern Matching

**Pattern Matching**이란 if then else 를 보다 쉽고 직관적으로 표현할 수 있는 방법이다. 위의 if then else 는 아래 `match with`와 동일하다.

```ocaml
# let ie = match tf with
    true -> 1
    | false -> 0;;

val ie : int = 1
```

`match` 키워드 바로 뒤에 오는 값과 `with` 키워드 뒤의 값들을 비교하여 적절한 값을 찾는 것으로 이해할 수 있다.

```ocaml
# let mat = match int_num with
10 -> true
| 20 -> false;;

Characters 10-53:
Warning 8: this pattern-matching is not exhaustive.
Here is an example of a case that is not matched:
0
val mat : bool = true
```

`match with`와 관련하여 조심해야 할 것이 있다면, 데이터 타입의 가능한 모든 경우의 수에 대해 어떻게 할 지 with 키워드 뒤에 정의해주어야 한다는 것이다. 위의 경우 integer type을 가지는 int_num의 가능한 경우의 수는 모든 정수인데, 10, 20 두 가지에 대해서만 정의해서 에러가 발생했다.

아래와 같이 underbar`_` 를 이용하면 해결할 수 있다.

```ocaml
# let mat = match int_num with
10 -> true
| _ -> false;;

val mat : bool = true
```

---

## 6. Function Declaration

ocaml에서는 함수 또한 `let` 키워드를 사용하여 정의한다. 예를 들어 입력으로 받은 정수에 10을 더해 반환하는 함수 `add_10`은 다음과 같이 정의할 수 있다.

```ocaml
# let add_10 a = a + 10;;

val add_10 : int -> int = <fun>
```

정의의 결과를 보면 `let a = 10`와 크게 차이가 없다. 단지 자료형이 `int -> int`와 같이 화살표로 표현된다는 점이 특이할 뿐이다. 함수형 프로그래밍에서는 함수도 1급 객체로 본다는 점에서 다른 자료형과 동일하게 취급된다는 점을 확인할 수 있는 부분이다.

함수 또한 자료형을 명시할 수 있다.

```ocaml
# let add_10 : int -> int = fun a -> a + 10;;

val add_10 : int -> int = <fun>
```

그 실행은 다음과 같다.

```ocaml
# add_10 20;;

- : int = 30
```

20을 입력으로 넣었더니 30을 받았다.

```ocaml
 # let sum a b = a + b;;
val sum : int -> int -> int = <fun>

# sum 10 20;;
- : int = 30
```

다른 언어와 차이 중 하나는 함수의 파라미터를 전달할 때 소괄호 `()`, comma `,` 등을 사용하지 않는다는 것이다.

---

## 7. Recursion not For/While

ocaml에서는 for/while과 같이 다른 언어에서 사용하는 반복문들을 사용하지 않는다. 대신 `rec` 키워드로 재귀 함수를 만들어 사용하는 것을 권장한다.

대표적인 재귀 함수를 사용하여 구현하는 factorial을 계산하는 함수는 다음과 같이 정의된다.

```ocaml
# let rec fac n =
if n <= 0 then 1
else n * fac (n-1);;

val fac : int -> int = <fun>
```

그 결과는 다음과 같다.

```ocaml
# let result = fac 5;;

val result : int = 120
```

당연하지만 재귀가 이뤄지므로 출구가 반드시 필요하다. 함수 내에서 현재 동작 중인 함수를 재귖거으로 사용하기 위해서는 recursion를 의미하는 `rec` 를 반드시 let 키워드 뒤에 붙여주어야 한다.
