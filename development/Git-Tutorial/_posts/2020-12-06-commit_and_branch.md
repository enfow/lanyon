---
layout: post
title: Commit & Branch
category_num : 1
---

# Commit & Branch

- Updated at: 2020.12.06
- [git](<https://git-scm.com/>)
- [learn git branching](<https://learngitbranching.js.org/?locale=ko>)

## Summary

- **Commit** is Snapshot
- **HEAD** is Working Position
- **Branch** is Pointer

## Introduction

[Git](<https://git-scm.com/>) 홈페이지에서는 Git을 다음과 같이 정의한다.

- Git is a free and open source **distributed version control system** designed to handle everything from small to very large projects with speed and efficiency.

Git은 **분산 버전 관리 시스템**이라고 한다. 여기서 **버전**이 작업하고 있는 코드의 상태를 의미하는데, 이러한 점에서 Git은 개발 과정에서 지속적으로 변경되는 코드를 관리해주는 프로그램이라고 할 수 있다. 그리고 **분산**이라는 표현에서 강조하고 있듯이 Git에서는 복수의 사람들이 작업하는 상황에서도 누가 어떤 코드를 언제 작성했는지 확인하고, 서로 다른 개발자의 코드를 일정한 규칙에 따라 하나로 합쳐주는 기능을 제공한다.

## Commit is SnapShot

- [git-commit](<https://git-scm.com/docs/git-commit>)
- [git-Recording Changes to the Repository](<https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository>)
- [git Tagging](<https://git-scm.com/book/en/v2/Git-Basics-Tagging>)
- [Commit Message Convention](<https://chris.beams.io/posts/git-commit/>)

**커밋(Commit)**은 코드 관리의 가장 기본적인 단위로, 특정 파일(혹은 모든 파일)에 대한 **스냅샷**이라고 할 수 있다. 즉 커밋으로 기록이 되어야만 Git에서 해당 코드를 관리하게 된다.

```bash
$ git add <file_name>
$ git commit
```

위와 같이 커밋에 담고자 하는 file을 `git add` 명령어로 선택하고 `git commit`명령어를 입력하여 특정 파일에 대한 커밋을 만들 수 있다. 이때 커밋을 만들었다는 것은 기존 코드와 비교하여 변경 내용이 생겼다는 것을 의미한다. 앞으로 커밋을 단위로 코드를 관리하게 될 것이므로 해당 Commit에 어떤 내용이 담겨 있는지 알려주는 메시지를 남겨 두어야 한다. 혼자 작업하고 있다면 자신만의 방식으로 작성해도 되나 다른 사람과 협업을 하고 있다면 [Commit Message Convention](<https://chris.beams.io/posts/git-commit/>)에 따라 메시지를 작성해주는 것이 좋다. (모든 Convention이 그렇듯 반드시 지켜야 하는 것은 아니며, 팀원 간에 상의하여 결정하는 것이 가장 좋다)

### Commit Stores Delta

`git commit`을 하게 되면 **커밋 트리(Commit Tree)**가 아래 그림과 같이 업데이트된다.

<img src="{{site.image_url}}/development/git_commit.png" style="width:30em; display: block;  margin: 0em auto;">

기존 `C1`에서 새로운 커밋 `C2`가 생겼다. 위에서 커밋은 스냅샷이라고 했는데, 이때 관리하는 모든 코드를 개별 커밋에 담게 되면 `C1`과 `C2`에는 중복이 존재할 수밖에 없다. Git에서는 이러한 중복을 줄이기 위해 전체 코드를 모두 커밋에 담지 않는다. 대신 개별 커밋에는 이전 커밋과 비교하여 다른 점(**Delta**)만을 기록해두고 변경이 없는 부분은 부모 커밋을 참조하도록 한다. 이러한 점 때문에 커밋 트리는 우측의 파란 화살표와 같이 자식 커밋 `C2`에서 부모 커밋`C1`을 가리키도록 그리게 된다.

### Start Point is HEAD

**HEAD**는 커밋 트리에서 현재 작업 중인 위치를 나타낸다. 따라서 `git commit`으로 새로운 커밋을 생성하게 되면 기존 HEAD의 위치에 따라 다른 점을 찾아내어 이를 커밋에 담게 된다. 커밋을 생성하면 HEAD 또한 새로운 커밋으로 이동한다. 위의 그림에서는 브랜치(Branch) 이름에 붙어 있는 `*`로 `Master`가 HEAD의 위치라는 것을 표현하고 있다.

<img src="{{site.image_url}}/development/git_commit_head.png" style="width:30em; display: block;  margin: 0em auto;">

위의 그림과 같이 브랜치의 마지막과 HEAD가 분리되어 있다면 HEAD의 위치가 가지는 의미를 보다 쉽게 확인할 수 있다.

### Tag the Commit

Git을 사용하다보면 과거의 어떤 커밋을 특정해야 하는 경우가 종종 발생한다. 모든 커밋은 고유의 키값을 가지고 있어 이를 통해 특정하는 것이 가능하지만 매번 커밋의 기값을 확인하는 것은 쉽지 않다. 대신 Git에서는 주요 커밋에 대해서 꼬리표(Tag)를 붙일 수 있도록 하고 있다. 꼬리표를 붙이는 방법은 아래와 같다.

```bash
git tag <tag_name> <commit_name> 
```

### Command for Commit

커밋과 관련된 명령어로는 다음과 같은 것들이 있다.

| Command |  Meaning  |
|---|---|
| `git status` | 현재 파일들의 상태를 확인한다 |
| `git add <file_name>` | 특정 파일을 stage에 올린다 |
| `git diff` | 최근 커밋과의 차이를 확인한다 |
| `git diff --staged` | stage에 올라온 파일에 대하서만 최근 커밋과의 차이를 확인한다 |
| `git commit` | stage에 올라와 있는 파일들의 변경 사항을 모아 커밋으로 만든다 |
| `git commit -m <msg>` | 커밋으로 만들 때 메시지를 바로 쓸 수 있도록 하는 옵션 |
| `git commit --amend` | 현재 커밋을 변경하도록 하는 옵션, 브랜치의 끝에 위치한 커밋에서만 가능 |
| `git log` | 최근 커밋을 확인한다. |
| `git log --oneline` | 커밋에 대한 정보들을 한 줄로 보여준다. |
| `git log --decorate` | HEAD, 브랜치 등을 확인할 수 있다. |
| `git log --graph` | 그래프 형태로 커밋들을 확인할 수 있다. |
| `git tag` | 테그 리스트를 확인한다 |

## Branch is Pointer

- [git-Branching](<https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>)
- [git-Branch Management](<https://git-scm.com/book/en/v2/Git-Branching-Branch-Management>)
- [MS-Adopt a Git branching strategy](<https://docs.microsoft.com/en-us/azure/devops/repos/git/git-branching-guidance?view=azure-devops>)

Git에서 **브랜치(Branch)**란 개별 작업이 진행 중인 위치를 가리키는 포인터라고 할 수 있다. 브랜치가 있기 때문에 여러 사람들이 각자 작업한 코드를 관리하고 이를 통합할 수 있다는 점에서 Git의 가장 핵심적인 기능이다.

브랜치의 목적 자체가 개인 단위 또는 기능 단위 작업 위치를 특정하는 것인 만큼 브랜치 또한 그에 맞춰 생성해주는 것이 좋다. 브랜치에 대한 관리 비용은 거의 들이 않으므로 최대한 잘게 쪼개는 것을 좋아하는 개발자들도 있다고 한다. 커밋 메시지처럼 브랜치 이름에 대해서도 컨벤션이 존재하는데 [이를](<https://docs.microsoft.com/en-us/azure/devops/repos/git/git-branching-guidance?view=azure-devops>) 확인해보면 브랜치의 단위에 대해 보다 쉽게 감을 잡을 수 있다.

### Create New Branch

브랜치를 새로 만드는 방법은 매우 직관적이고 간단하다.

```bash
$ git branch <branch_name>
```

아래 그림은 `<branch_name>`으로 feature를 입력했을 때 커밋 트리의 변화를 보여준다.

<img src="{{site.image_url}}/development/git_branch.png" style="width:25em; display: block;  margin: 0em auto;">

현재 HEAD의 위치(`*master`)에 새로운 브랜치 `feature`가 생성되었고, `master` 브랜치와 동일한 커밋을 가리키는 것을 알 수 있다.

<img src="{{site.image_url}}/development/git_branch_commit.png" style="width:25em; display: block;  margin: 0em auto;">

이때 새로운 커밋을 만들게 되면 HEAD가 가리키고 있는 `master` 브랜치는 업데이트되지만 `feature` 브랜치는 그대로 `C2` 커밋에 머물러 있게 된다.

### Move to another Branch

`feature` 브랜치에서 작업을 수행하고 싶다면 HEAD가 가리키는 브랜치를 바꿔주어야 한다. 작업 브랜치를 변경하는 명령어는 `git checkout` 이다.

```bash
$ git checkout <branch_name>
```

아래 그림을 보면 HEAD를 뜻하는 `*`가 `master`에서 `feature`로 이동했음을 알 수 잇다.

<img src="{{site.image_url}}/development/git_branch_checkout.png" style="width:25em; display: block;  margin: 0em auto;">

브랜치가 바뀐 상황에서 커밋을 하게 되면 `feature` 브랜치로 새로운 커밋이 추가된다.

<img src="{{site.image_url}}/development/git_branch_checkout_commit.png" style="width:32em; display: block;  margin: 0em auto;">

### Command for Branch

커밋과 관련된 명령어로는 다음과 같은 것들이 있다.

| Command |  Meaning  |
|---|---|
| `git branch` | 브랜치 리스트를 확인한다. |
| `git branch -v` | 각 브랜치를 마지막 커밋 메시지와 함께 보여준다. |
| `git branch -d` | 브랜치를 삭제한다. |
| `git branch -D` | merge되지 않았더라도 브랜치를 강제로 삭제한다. |
| `git branch <new_branch_name>` | 현재 HEAD를 기준으로 새로운 브랜치를 생성한다. |
| `git checkout <branch_name>` | 다른 브랜치로 HEAD를 이동한다. |
| `git checkout -b <new_branch_name>` | 새로운 브랜치를 생성하고 HEAD를 그에 맞춰 이동한다. |