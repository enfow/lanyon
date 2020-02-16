---
layout: post
title: C Data Type
category_num: 2
---

# C Data Type

- [C89 표준](<https://www.pdf-archive.com/2014/10/02/ansi-iso-9899-1990-1/ansi-iso-9899-1990-1.pdf>)을 최대한 참고하여 작성하였습니다.
- update at 2020.02.16

## Byte

자료형에 대한 정의에 앞서 C89 표준에서는 **Byte**를 어떻게 보고 있는지 확인할 필요가 있다.

- byte: The unit of data storage **large enough to hold any member of the basic character set of the execution environment.** It shall be possible to express the address of each individual byte of an object uniquely. A byte is **composed of a contiguous sequence of bits**, the number of which is implementation-defined.

C89 표준에서는 Byte의 명시적인 크기를 제시하지 않고 있다. 대신 **실행환경에서 기본적인 문자들을 표현할 수 있는 충분한 길이의 연속적인 비트열** 정도로 정의한다. 이를 만족하기 위해서 최소한 8bits 정도가 필요하기 때문에 **1 Byte = 8bits**가 된 것이다.

Byte 정의 바로 아래줄에서 character도 정의하고 있는데 그 내용은 다음과 같다.

- character: A bit representation that fits in a byte. The representation of each member of the basic character set in both the source and execution environments shall fit in a byte.

한마디로 **character**는 Byte와 동일한 크기를 갖는다고 한다. 이는 아래에서 정리하는 자료형 char의 특성이 된다.

## CHAR

### size of char

C89 표준에서는 **char**의 크기와 관련해 다음과 같이 정의내리고 있다.

- An object declared as type char is large enough to store any member of the basic execution character set.

즉 표준에서는 char의 크기를 명시적으로 8bit라고 정의하는 것이 아니라, **실행환경의 기본적인 문자 집합을 저장하기 충분한 크기**로만 정의한다. 이는 Byte의 정의와 동일하다. 즉, **Byte의 크기가 곧 char의 크기**가 되는데 Byte의 크기 또한 명시적이지 않으므로 실행 환경, 컴파일러 등에 따라 다르게 설정될 수 있다고 보아야 한다. 일반적인 데스크탑 환경에서는 1 Byte를 8bits로 본다.

### Signed char & Unsigned char

signed, unsigned란 부호의 유무와 관련된다. 첫 번째 비트를 부호로 사용하는 signed 가 기본이므로 생략되는 편이며, unsigned를 사용하면 음수가 없다는 것을 의미하므로 표현할 수 있는 양수의 범위가 2배 가까이 늘어난다.

다만 char의 경우에는 signed char, unsigned char 그리고 char 세 가지가 모두 다르다.

- char : 0 ~ 127
- signed char : -127 ~ 127
- unsigned char : 0 ~ 255

char의 경우 signed/unsigned가 없으면 첫 번째 비트는 사용하지 않는 것으로 보기 때문이다.

## INT

### size of int

#### Minimum size

- A plain int object has the natural size suggested by the architecture of the execution environment (large enough to contain any value in the range INT_MIN to INT_MAX as defined in the header `<limits.h>`).

정수를 표현하는 가장 일반적인 자료형 **int** 또한 char와 마찬가지로 명시적인 bit 수를 정의하지는 않고 있다. 하지만 header file `<limits.h>`에 명시된 INT_MIN, INT_MAX를 표현할 수 있어야 한다고 되어 있다. int 뿐만 아니라 다른 많은 C의 자료형이 정확한 bit 수가 아닌 표현해야 하는 숫자의 최소 범위를 제시하고 있다.

- INT_MIN : -32767
- INT_MAX : +32767
- UINT_MAX : 65535

$$2^{15} = 32768$$, $$2^{16} = 65536$$이라는 점을 감안할 때 int 자료형은 최소 16bits 이상이어야 한다.

#### size of int on modern architecture

위에서 보았듯이 int는 16bits(2 Byte) 이상이면 된다. 최근에 사용하는 int의 크기는 32bits(4Byte)인 경우가 많은데 이는 CPU의 발전과 관련 깊다. [stack overflow](<https://stackoverflow.com/questions/11438794/is-the-size-of-c-int-2-bytes-or-4-bytes>)에 따르면 C가 처음 만들어졌을 당시 연산의 기본단위는 16bits였고 이 때문에 int의 크기로 2byte를 주로 사용했지지만, CPU의 발전으로 연산의 기본단위가 32bits, 64bits로 커지면서 int의 크기도 함께 커졌다는 것이다. 현재 데스크탑 CPU의 표준인 x86의 경우 64bits를 기본 연산 단위로 사용하고 있다. 하지만 int의 경우 32bits 표준에서 더 이상 증가하지 않고 대다수의 환경에서 4Byte의 크기를 가진다.

## SHORT

### size of short

short의 최소 크기와 관련해 `<limits.h>`에서는 다음과 같이 정의하고 있다.

- SHRT_MIN : -32767
- SHRT_MAX : +32767
- USHRT_MAX : 65535

최소크기만 보면 short는 int와 동일하다. 즉 최소한 16bits 이상이어야 한다. modern architecture에서 int의 크기는 4Bytes로 커졌지만, short는 그 의미에 맞게 2Byte의 크기를 가지고 있다.

## LONG

### size of long

long의 최소 크기와 관련해 `<limits.h>`에서는 다음과 같이 정의하고 있다.

- LONG_MIN : -2147483647
- LONG_MAX : +2147483647
- ULONG_MAX : 4394967295

$$2^32=4294967296$$라는 점을 고려할 때 long의 크기는 최소한 4Byte가 되어야 한다. 이렇게 보면 int의 크기는 원래 short와 같았지만, 크기가 커지면서 long과 동일해졌다.

## FLOAT

### real number in computer system

**float**는 floating number의 준말로, 우리말로 하면 부동소수점이다. 즉 float는 소수점 아래의 숫자, 유리수를 다루기 위한 자료형이다.

부동소수점과 관련된 표준으로는 1885년 제정된 IEE 754가 있는데, 표준([wiki IEEE754](<https://en.wikipedia.org/wiki/IEEE_754>))에서는 32 bits 부동 소수점의 구조를 다음과 같이 표현하고 있다.

- sign : 1bit
- exponent : 8bits
- fraction : 23bits

### size of flaot

이에 따라 C에서도 일반적으로는 **32bits**를 float의 기본 크기로 하고 있다. 물론 float 또한 명시적인 크기는 없다.

## DOUBLE

### size of double

IEEE 754에서는 binary32를 single precision, binary64를 double precision이라고 한다. **double**은 binary64를 따르는 자료형이며, 64라는 표현에서 짐작할 수 있듯 **64bits**의 크기를 갖는다.

## Additional Study

- LLP64/IL32P64
