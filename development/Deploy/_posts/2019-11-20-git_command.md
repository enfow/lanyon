---
layout: post
title: Git commands
---

# 자주 쓰는 Git 코드 모음

- update at 19.11.20

### 기본적인 사용 방법

깃허브의 원격 저장소를 이용하여 작업할 때에는 다음과 같은 과정을 따른다.

 1. pull - 원격저장소에서 파일 가져오기
 2. 코드 작성하기
 3. add - 변경사항 중 커밋할 대상 스테이지에 올리기
 4. commit- 스테이지 상의 파일 커밋하기
 5. push - 원격 저장소에 파일 올리기

### 변경사항이 있는 파일 stage에 추가

- 모든 파일 추가

    `git add . `

- 특정 파일 /디렉토리 추가

    `git add <file/dir name>`

### stage에서 제거하기

- stage 상의 모든 파일 제거

    `git reset`

- stage 상의 특정 파일 제거

    `git reset <file name>`

### stage의 파일 commit

- 커밋

    `git commit -m <“commit message”>`

### 파일 상태 확인

- 변경된 파일과 현재 브랜치 확인

    `git status`

- 빨간색일 경우 아직 stage에 올라가지 않은 파일
- 녹색일 경우 stage에서 commit 대기 중인 파일

### 브랜치 합치기: merge

- 브랜치 병합

    `git merge <merge_대상_branch>`

- 이때 branch는 merge를 받는 branch여야 한다.

### 가져와서 합치기: pull

- pull = fetch + merge

- 로컬 브랜치에서 가져와 합치기

    `git pull <branch_name>`

- 원격 브랜치에서 가져와 합치기

    `git pull origin <branch_name>`

- pull은 merge와 fetch 를 합친 것이다. 즉 fetch로 원격에서 가져와 merge로 병합하는 것이 가능하다. pull로 한 번에 하는 것보다 fetch와 merge 두 개로 나누어 하는 것이 더 안전한 방법이다.

- ex)
  - git pull origin master : 원격의 master 브랜치에서 가져온다.
  - git pull origin develop_runner : 원격의 origin/develop_runner 브랜치에서 현재 브랜치로 가져와 병합한다.

### 보내서 업데이트하기: push

- 로컬 브랜치 업데이트하기

    `git push <branch_name>`

- 원격 브랜치 업데이트하기

    `git push origin <branch_name>`

### 브랜치 확인

- 로컬 브랜치 확인

    `git branch`

- 원격 브랜치 확인

    `git branch -r`

### 브랜치 생성

- 새로운 브랜치 생성

    `git branch <new_branch_name>`

### 브랜치 변경

- 기존 브랜치로 이동

    `git checkout <branch_name>`

- 새 브랜치 생성 후 이동

    ``git checkout -b <new_branch_name>``


### 로컬과 리모트

깃은 로컬과 리모트로 나누어진다. 만약 clone을 받아 remote에서 작업하고 싶다면 다음과 같이 작업하면 된다.

 1. 로컬 브랜치 생성

    `git branch <new_branch_name>`

 2. 로컬 브랜치에 commit 할 게 있다면 commit 완료하기

    `git add .`

    `git commit -m <“commit message”>`

 3. remote에 push 하기

    `git push origin <branch name>`

    - 이때 현재 로컬 브랜치 명과 일치하는 브랜치가 없다면 새로 생성된다.

### 로컬 깃을 깃허브와 연결하기

```
$ git init
$ git config --global user.name "Your Name"
$ git config --global user.email you@example.com

$ git add .
$ git commit -m “”

$ git remote add origin <git hub http>

$ git push -u origin master
```

### 브랜치 충돌

- 로컬 브랜치 간 비교

    `git diff <branch_name> <branch_name>`

- 로컬 - 원격 브랜치 간 비교

    `git diff<branch_name> origin/<branch_name>`

### 충돌의 해결

- 원격에서 pull 받을 때 conflict가 예상되어 Aborting되는 경우

    - commit 되지 않은 파일 중 conflict가 발생하는 경우이다. 아래와 같이 해결한다.
    1. git stash로 충돌 예상 파일 숨기기
    2. pull을 받는다.
    3. git stash pop으로 변경 파일을 복구한다.
    4. 복구하게 되면 충돌 예상 파일에 conflict log가 뜬다.
    5. conflict를 해결한다.
    6. 해결 후 해당 파일을 commit하면 완료

### 변경 파일 숨기기: stash

- 현재 branch의 변경사항을 일시적으로 숨김

    `git stash`

- 숨김 해제

    `git stash pop`

- 현재 branch에서 변경된 것이 있으면 다른 branch로 checkout 할 때 Aborting error가 발생한다. 브랜치를 착각하여 잘못된 브랜치에서 작업한 경우도 종종 있다. 이 경우 현재 작업 내용을 숨기고 이동하여 작업을 계속해야하는데 stash를 사용하면 간편하게 해결할 수 있다.

### 원격 브랜치 가져오기

- git pull origin master를 하면 현재 브랜치와 연결된 브랜치의 코드만 가져온다. 따라서 로컬에 없는 브랜치를 원격으로부터 가져오기 위해서는 따로 가져오는 작업이 필요하다.

- 로컬 브랜치 확인

    `git branch`

- 원격 브랜치 확인

    `git branch -r`

- 로컬 + 원격 브랜치 확인

    `git branch -a`

- 원격 브랜치 가져오기

    `git checkout -b origin/<remote_branch_name>`

- ex)
    - 원격의 develop/model 브랜치를 가져오고 싶은 경우 -> git checkout -b origin/develop/model  

### commit 취소하기

- commit history 확인하기

    `git log`

- commit 취소하기

    `git reset HEAD^`

- commit 취소하기 + 취소된 commit을 stage에 올린 상태로 전환하기

    `git reset --soft HEAD^`
