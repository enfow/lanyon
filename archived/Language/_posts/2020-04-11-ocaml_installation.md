---
layout: post
title: Ocaml Installation
category_num: 1
---

# Ocaml Installation

- update date : 2020.04.11
- [Install OCAML](<https://ocaml.org/docs/install.html>)

## MAC

ocaml과 ocaml에서 패키지 등을 관리해주는 opam을 설치해준다.

```
$ brew install ocaml
$ brew install opam
$ eval `opam config env`
```

opam은 python의 pip와 유사한 방식으로 동작한다. 예를 들어 utop 이라는 ocaml REPL(read-eval-print loop)를 설치한다고 한다면 다음과 같이 하면 된다.

```
$ opam install utop
```

설치를 확인하기 위해 다음과 같이 실행할 수 있다.

```
$ utop
```

기본으로 설치되는 occaml REPL의 경우 사용상 불편한 점이 많아 [utop](<https://opam.ocaml.org/blog/about-utop/>) 을 사용하는 것을 추천한다.

## UBUNTU

ubuntu에서도 mac에서와 크게 다를 것 없이 간단하게 설치할 수 있다.

```
$ apt install ocaml
$ apt install opam
$ eval $(opam env)
```
