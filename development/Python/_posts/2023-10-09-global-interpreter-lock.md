---
layout: post
title: Python Global Interpreter Lock
category_num: 3
---

# Python Global Interpreter Lock

- update date : 2023.10.09

## GIL과 Python Memory Management

파이썬 GIL 이란 **"한 프로세스 내에서 파이썬 바이트코드가 하나의 쓰레드에서만 실행될 수 있도록 하는 mutex(mutual exclusive)"**를 말한다. 즉 하나의 파이썬 프로세스 내에 여러 개의 쓰레드가 존재하더라도 각각의 쓰레드는 GIL을 얻은 상태일 때에만 실행될 수 있다. CPU에 아무리 많은 코어가 존재하더라도(Multi prossessor) 하나의 파이썬 쓰레드만 처리될 수 있기 때문에 파이썬에서는 멀티 쓰레딩으로 인한 이점이 제한된다.

```
this mutex is necessary mainly because CPython's memory management is not thread-safe.

https://wiki.python.org/moin/GlobalInterpreterLock
```

 python wiki 에서는 GIL이 존재하는 이유를 파이썬의 메모리 관리가 thread-safe 하지 않다는 점에서 찾고 있다. 보다 깊이 들어가면, Python Memory Manager는 모든 파이썬 객체들의 reference count 를 가지고 있으며, 이를 기준으로 garbage collection 을 수행한다. 

 그런데 병렬로 여러 개의 Thread 가 실행되면 race condition의 상황이 발생하여 이 reference count 가 정확히 계산되지 않을 수 있다. 물론 개별 reference count 에 mutex를 건다면 해결할 수 있지만, 매우 비효율적이므로 실행 단위인 쓰레드를 단위로 Lock 을 만든 것이다.

## Reference

- [Python wiki](<https://wiki.python.org/moin/GlobalInterpreterLock>)
- [Python dev guide](<https://devguide.python.org/internals/garbage-collector/>)