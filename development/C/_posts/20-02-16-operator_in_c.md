---
layout: post
title: operator in C
category_num: 3
---

# Operator in C

- [C89 표준](<https://www.pdf-archive.com/2014/10/02/ansi-iso-9899-1990-1/ansi-iso-9899-1990-1.pdf>)을 최대한 참고하여 작성하였습니다.
- update at 2020.02.16

## 1. sizeof

**sizeof는 함수가 아니라 연산자(operator)이다.** 이와 관련하여 C89에서는 다음과 같이 정의하고 있다.

- The sizeof **operator** yields **the size (in bytes) of its operand.** which may be an expression or the parenthesized name of a type.

즉 sizeof 연산자는 피연산자의 크기를 byte 단위로 반환한다. char의 경우 Byte와 같은 크기로 정의되므로, 항상 그 값이 1이 되고, 다른 자료형의 경우에는 환경(16bits, 32bits, 64bits)에 따라 byte 크기가 달라질 수 있으므로 환경에 따라 달리 반환될 수 있다.

### size_t

sizeof의 반환값은 int가 아닌 **size_t**형 인데, 이는 `<stddef.h>` header file에 정의되어 있으며, unsigned int와 크기가 동일하다. 무언가의 크기를 표현할 때 사용되는 자료형이므로 unsigned이어야 한다. 그리고 for loop, malloc의 파라미터와 같이 음수가 오면 안되는 경우에 많이 사용한다.

표준에서는 size_t의 정확한 크기에 대해 언급하고 있지 않은데, [stack overflow](<https://stackoverflow.com/questions/918787/whats-sizeofsize-t-on-32-bit-vs-the-various-64-bit-data-models>)와 [quora](<https://www.quora.com/What-is-size_t-in-C-programming>)에 따르면 32bit 시스템에서는 unsigned int와 같은 4 Byte, 64bit 시스템에서는 8 Byte의 크기를 가진다.
