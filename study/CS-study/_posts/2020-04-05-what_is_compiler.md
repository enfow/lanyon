---
layout: post
title: (Compiler) What is Compiler
category_num: 200
---

# What is Compiler

- update date : 2020.04.05

## Introduction

**Compiler**란 **어떤 프로그램을 입력으로 받아 다른 언어로 작성된 동일한 프로그램을 반환하는 프로그램**으로, 쉽게 말해 프로그램 번역기라고 할 수 있다. 일반적으로 컴파일러를 말한다면 C, C++, JAVA 등의 High Level Language(HLL)로 작성된 프로그램을 x86, arm 과 같은 Low Level Language로 변환하여 프로세서가 이해할 수 있는 machine code로 변환해주는 프로그램을 의미한다.

## Compiler Structure

컴파일러는 크게 FrontEnd, MiddleEnd, BackEnd 라고 불리는 세 가지 부분으로 이루어져 있으며, 각각의 부분들은 또다시 여러가지 하위 요소들을 가지고 있다.

- input : Source Language Program
- output : Target Language Program with the same meaning as input

위에서 설명한 것과 같이 Compiler는 어떤 언어로 작성된 프로그램을 다른 언어로 작성된 동일한 프로그램으로 변환해주는데, 입력으로 들어오는 프로그램의 언어를 Source Language라고 하고, 출력으로 나가는 프로그램의 언어를 Target Language 라고 한다. 이때 중요한 특징 중 하나는 입력 프로그램과 출력 프로그램의 언어는 서로 다르지만 그 의미는 동일해야 한다는 것이다.

## 1. FrontEnd

컴파일러 프론트엔드의 역할은 입력으로 들어온 프로그램에 문법적/의미적 문제가 없는지 검사하고, 미들엔드와 백엔드에서 사용하기 쉽도록 구조화하여 전달하는 역할을 갖는다. 구체적으로 프론트엔드의 출력물인 **Intermediate Representation(IR)**은 그 하위 부분인 Lexical Analyzer, Syntax Analyzer, Semantic Analyzer, IR translator 를 순차적으로 통과한 결과라고 할 수 있다.

---

### 1.1. Lexical Analyzer

- input : Source Language Program
- output : Token Stream

**Lexical Analyzer**는 줄여서 **lexer**라고도 부르는데, 프론트엔드에서도 가장 첫 부분으로 입력 프로그램을 받아서 의미적으로 구성된 토큰으로 변환하는 작업을 수행한다. 문법적, 의미적으로 정확한 프로그램을 작성했다고 하더라도 컴퓨터가 보기에는 문자열에 불과하다. 이러한 문자열을 해당 언어의 문법에 맞게 의미를 갖는 개별 토큰으로 분리하는 과정이라고 할 수 있다.

---

### 1.2. Syntax Analyzer

- input : Token Stream
- output : Syntax Tree

**Syntax Analyzer**는 문법적인 구조를 분석하는 단계로, lexical analysis가 단어를 다루었다면 syntax analyzer는 문장을 다루는 과정이라고 할 수 있다. 입력으로 들어온 Token Stream을 의미에 맞게 **Tree** 구조로 Parsing 하여 다음 단계로 전달하게 된다. 이러한 점에서 **Parser**라고도 부른다.

---

### 1.3. Semantic Analyzer

- input : Syntax Tree
- output : Syntax Tree

**Semantic Analyzer**는 입력과 출력이 모두 Syntax tree 로 동일한데, 여기서는 프로그램의 의미(semantic)를 확인하는 작업을 수행한다. 즉, syntax analyzer 에서는 구조를 분석했다면, semantic analyzer 에서는 해당 구조의 의미가 가능한지 확인하는 것이라고 할 수 있다. 만약 의미적으로 부적절한 요소가 있다면 에러를 출력하게 되는데, 대표적으로 type error, array out of bounds, null-dereference, divide by zero 등이 있다.

Sementic analyzer 를 강화하여 가능한 모든 문법적, 의미적 오류들을 걸러낼 수도 있다. 하지만 비용을 고려하여 일반적으로는 반드시 필요한 문제들만 확인한다.

---

### 1.4. IR translator

- input : Syntax Tree
- output : IR

프론트엔드의 마지막 단계로 **IR translator**은 말 그대로 Synrax Tree를 번역하여 IR로 변환한다. **Intermediate Representation(IR)**은 말 그대로 중간 언어로 작성된 프로그램으로, 이 또한 컴파일러의 종류에 따라 다양한 형태가 존재한다. 대표적으로 LLVM 등에서 사용하는 [3-address code](<https://ko.wikipedia.org/wiki/3-%EC%96%B4%EB%93%9C%EB%A0%88%EC%8A%A4_%EC%BD%94%EB%93%9C>)가 있다.

이렇게 source language에서 target language로 바로 바꾸지 않고, 중간 단계를 거치는 이유는 컴파일러의 개발을 편리하게 하기 위함이다. 예를 들어 source language가 $$N$$개 존재하고, target lanauge는 $$M$$개 존재하는 상황에서 모든 경우의 수를 만족하기 위해서는 총 $$N \cdot M$$ 개의 컴파일러를 개발해야 한다. 반면 중간 단계인 **IR**을 도입하면 source lanaguage를 IR 로 바꾸는 프론트엔드와 IR을 target language로 바꾸는 백엔드를 각각 만들고 서로 조합하기만 하면 되므로 $$N + M$$ 개만 만들면 되기 때문에 보다 효율적이다.

---

## 2. MiddleEnd

- input : IR
- output : optimzized IR

미들엔드는 **IR**로 작성된 프로그램에 대해 최적화를 수행하는 단계로, 다수의 **optimzer**로 구성되어 있다. 최적화 단계이므로 반드시 필요한 과정은 아니며, optimizer의 포함 여부는 효율성을 고려하여 정책적으로 결정된다.

## 3. BackEnd

- input : IR
- output : Target Language Program with the same meaning as input

백엔드는 IR을 전달받아 Target Language로 변환하는 과정이라고 할 수 있다. 일반적인 컴파일러는 High Level Language를 Machine language로 변환하는 기능을 하게 되며, 백엔드에서는 Register의 크기와 같이 Machine language의 특유의 제약 사항을 고려하여 변환하는 것이 주된 문제가 된다고 할 수 있다.
