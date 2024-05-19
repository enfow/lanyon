---
layout: post
title: Keywords for OOP
category_num: 502
keyword: "[Programming]"
---

# Keywords for OOP

- 책 **GOF의 디자인 패턴(Design Patterns: Elements of Reusable Object-Oriented Softward)**의 Chapter 1에 나오는 키워드를 정리하고 있습니다.
- Update at: 2020.11.15

## Keywords List

- Object Oriented Programming(OOP)
  - Class, Instance and Object
- Inheritance & Polymorphism
  - Signature, Interface and Type
  - Difference between Class and Type
  - Parent Class and Child Class
  - Override
  - Abstract class & Concrete Class
  - Class Inheritance & Interface Inheritance

## Object Oriented Programming(OOP)

Wiki에서는 **객체 지향 프로그래밍(Object Oriented Programming, OOP)**를 다음과 같이 정의한다.

- **Object-oriented programming(OOP)** is a programming paradigm based on the concept of **"objects"**, which can **contain data and code**: data in the form of fields(often known as **attributes or properties**), and code, in the form of procedures(often known as **methods**).

이름에도 그대로 드러나듯 객체 지향 프로그래밍은 객체(Object)라고 하는 단위를 중심으로 하는 개발 패러다임이다. 객체 지향 프로그래밍과 대비되는 예시로 자주 사용되는 **절차적 프로그래밍(Procedural Programming)**에서는 프로그램을 절차, 즉 순차적인 명령어의 처리로 보는 반면 객체 지향 프로그래밍은 프로그램을 여러 객체들이 존재하고 각각이 상호작용하는 것으로 본다고 할 수 있다.

### Class, Instance and Object

그렇다면 **객체(Object)**란 무엇일까. 객체는 **클래스(Class)**, **인스턴스(Instance)**와 함께 설명하는 경우가 많은데, Wiki에서 찾을 수 있는 다음 정의들과 같이 이들은 밀접한 관계를 가지고 있기 때문이다.

- **Class**: An extensible program-code-template for creating objects, providing initial values for state and implementations of behavior
- **Instance**: A concrete occurrence of any **object**, existing usually during the runtime of a computer program
- **Object**: An **object** can be a variable, a data structure, a function, or a method, and as such, is a value in memory referenced by an identifier.

비유적으로 표현하면 클래스는 설계도, 인스턴스는 설계도로 만든 실제 세계에 존재하는 특정 제품, 객체는 실제 세계의 모든 제품을 말한다. 이때의 실제 세계란 하드웨어 적으로는 메모리를 뜻하고, 소프트웨어 적으로는 런타임을 의미한다. 인스턴스와 객체는 혼용되어 사용되는 경우도 많고, 이에 대한 글들을 찾아보면 설명 방식마다 약간의 차이가 있음을 쉽게 확인할 수 있는 만큼 사람마다 이해 방법이 조금 다른 것 같다. 개인적으로는 인스턴스와 객체 모두 메모리 상의 대상을 지칭하는데 인스턴스는 '인스턴스화'라는 표현처럼 어떤 클래스를 구현하였다는 것을 강조하는 표현이고, 객체는 좀 더 추상적인 개념으로 프로그램의 구성요소 혹은 일종의 단위에 가까운 표현이라고 생각한다.

## Inheritance & Polymorphism

### Signature, Interface and Type

모든 객체는 데이터(Property, Attribute, Member Variable)와 그 데이터에 대한 연산(Method)으로 구성된다. **Signature**란 Method의 특성(Method의 이름, Method의 Paramter, Method의 Return)들을 말한다. 그리고 어떤 객체의 **Interface**란 객체가 가지는 모든 Method의 Signature들의 집합이라고 할 수 있다. 어떤 객체가 가지는 **타입**이란 이러한 Interface를 기준으로 정해진다. 즉 타입이 같다는 것은 특정 Interface에서 정의하는 모든 Method들을 처리할 수 있다는 것을 의미한다.

여기서 클래스와 타입의 차이를 명확히 하고 갈 필요가 있는데, **객체의 클래스**라는 것은 위에서도 언급하였듯 설계도로서 객체가 가지는 Method의 구체적인 구현 방식에 대한 정보를 포함한다. 반면 **객체의 타입**은 Method의 구체적인 구현 방식에는 관심이 없고, Interface에만 집중한다. 따라서 하나의 객체가 복수의 타입을 가질 수 있고, 서로 다른 클래스에 따라 정의된 객체들이라고 할지라도 동일한 타입을 가질 수도 있다

### Inheritance & Override

**클래스 상속(Class Inheritance)**은 새로운 클래스를 정의할 때 미리 정의된 다른 클래스의 특성을 그대로 이어받는 것을 말한다. 이때 기존의 특성을 물려주는 클래스를 **부모 클래스(Parent Class)**라고 하고, 특성을 이어 받는 클래스를 **자식 클래스(Child Class)**라고 한다. 이때 자식 클래스에서 새로운 Property나 Method를 추가로 정의하지 않는다면 자식 클래스와 부모 클래스의 객체는 동일한 Interface를 가진다고 할 수 있다.

이때 자식 클래스에서 부모 클래스에 미리 정의된 Method와 동일한 Signature를 가지는 새로운 Method를 재정의할 수도 있는데, 이를 **오버라이드(Override)**라고 한다.

### Polymorphism

OOP에서 **다형성(Polymorphism)**이란 동일한 인터페이스를 가지지만 각 Method의 구현은 서로 다른 특성을 의미한다. 어떤 두 객체의 인터페이스가 동일하다면 각 Method의 구현이 완전히 다르다 하더라도 두 객체 모두 동일한 Request를 처리할 수 있다.

이와 같은 대체 가능성을 다형성이라고 하는 것이다. 자식 클래스에서 오버라이드를 하게 되면 구현은 다르지만 동일한 인터페이스를 가지는 객체를 생성하는 것이 가능한데, 이것이 다형성의 대표적인 예시이다. 포스팅 마지막에 있는 예시 코드는 Python에서 클래스 상속과 오버라이드를 통해 동일한 Interface를 가지지만 서로 다른 결과를 반환하는 경우를 보여주고 있다. parent, child 두 인스턴스의 Inferface가 동일하므로 동일한 Request에 대해 결과는 다르지만 동작하는 것을 알 수 있다.

### Abstract class & Concrete Class

**Abstract Class**란 상속받는 자식 클래스들이 공통으로 가져야하는 인터페이스를 정의하는 클래스를 말한다. 반대로 **Concrete Class**는 Abstract Class를 상속받아 구체적인 Method를 정의하는 클래스를 말한다. Python에서는 Abstact Class 를 정의할 때 `NotImplementedError()`를 사용하면 동일한 인터페이스를 구현하도록 강제할 수 있다.

### Class Inheritance & Interface Inheritance

상속은 크게 부모 클래스의 구체적인 구현 내용을 모두 가져오는 **클래스 상속**과 부모 클래스의 인터페이스만 가져오고 구현은 자식 클래스에서 하게 되는 **인터페이스 상속** 두 가지로 나누어 볼 수 있다. 쉽게 말해 Concrete Class를 상속받는 것은 클래스 상속, Abstract Class를 상속받는 것을 인터페이스 상속이라고 한다. 참고로 인터페이스 상속은 서브타이핑이라고도 한다.

Python을 포함한 대부분의 프로그래밍 언어가 두 가지를 정확하게 구분하지 않지만 클래스 상속보다 인터페이스 상속이 다음 두 가지 이유에서 더 좋다고 한다.

- 복수의 자식 클래스가 존재하더라도 모두 부모 클래스의 인터페이스를 따른다면 쉽게 다룰 수 있다.
- 구현 내용이 모두 자식 클래스에 있으므로 동작 방식을 이해하기 위해 부모 클래스를 확인할 필요가 없다.

즉, 유사한 목적을 가지는 복수의 클래스를 관리한다면 추상 클래스를 하나 정의하고 이를 상속하여 인터페이스는 맞추면서 구현 내용은 개별 자식 클래스에서 관리하도록 하자는 것이다. 이러한 관점에서 GOF의 디자인 패턴에서는 다음과 같은 표현이 나온다.

- 구현이 아닌 인터페이스에 따라 프로그래밍합니다.

## Example Codes

### Polymorphism

```python
  1 """Show Polymorphism Example."""
  2
  3
  4 class ParentClass:
  5     """Define ParentClass."""
  6
  7     def __init__(self) -> None:
  8         """Initialize."""
  9         self.name: str = "Parent"
 10
 11     def print_name(self) -> None:
 12         """Print name property."""
 13         print(self.name)
 14
 15     def print_number(self, inp: int) -> None:
 16         """Print input * 2."""
 17         print(inp * 2)
 18
 19
 20 class ChildClass(ParentClass):
 21     """Define ChildClass That inherit ParentClass."""
 22
 23     def __init__(self) -> None:
 24         """Initialize."""
 25         super().__init__()
 26         self.name: str = "Child"
 27
 28     # Method Override
 29     def print_number(self, inp: int) -> None:
 30         """Print input * -2."""
 31         print(inp * -2)
 32
 33
 34 if __name__ == "__main__":
 35     # Creat Instance
 36     parent = ParentClass()
 37     child = ChildClass()
 38
 39     for obj in (parent, child):
 40         obj.print_name()
 41         obj.print_number(10)
 42
 43     # ---RESULTS---
 44     # Parent
 45     # 20
 46     # Child
 47     # -20
```

### Abstract class & Concrete Class

```python
  1 """Show Abstract Class and Concrete Class Example."""
  2
  3
  4 class AbstractClass():
  5     """Define Abstract Class."""
  6
  7     def __init__(self) -> None:
  8         """Initialize."""
  9         self.name = "Abstract"
 10
 11     def print_name(self) -> None:
 12         """Print name of the object."""
 13         raise NotImplementedError()
 14
 15     def print_star(self) -> None:
 16         """Print star *."""
 17         raise NotImplementedError()
 18
 19
 20 class ConcreteClass(AbstractClass):
 21     """Define Concrete Class."""
 22
 23     def __init__(self) -> None:
 24         """Initialize."""
 25         super().__init__()
 26         self.name = "Concrete"
 27
 28     # Override
 29     def print_name(self) -> None:
 30         """Print name of the object."""
 31         print(self.name)
 32
 33     # Override
 34     def print_star(self) -> None:
 35         """Print star *."""
 36         print("*")
 37
 38
 39 if __name__ == "__main__":
 40     obj = ConcreteClass()
 41     obj.print_name()
 42     obj.print_star()
 43     # ---RESULTS---
 44     # Concrete
 45     # *
```
