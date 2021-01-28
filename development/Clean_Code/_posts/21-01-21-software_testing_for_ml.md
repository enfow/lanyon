---
layout: post
title: "Testings for Machine Learning: Introduction"
category_num : 1
---

# Testings for Machine Learning: Introduction

- 머신러닝 리서치 엔지니어로 일하면서 느낀 Software Testing의 필요성에 대해 정리해보았습니다.
- Update at: 2021.01.21

## Why Software Testing?

Software Testing이란 작성한 코드가 원하는대로 정확히 동작하는지 검증하는 작업을 말하는데, 최근에는 PyTest와 같은 Test Tool을 사용하여 자동화하는 것이 일반적입니다. 이렇게 Test를 자동화하게 되면 어떤 함수에서 어떤 문제가 있는지 한 번에 알 수 있으면서, Test를 실행하는 것이 명령어 한 줄로 간단해져 자주 확인하게 된다는 장점이 있습니다. 또한 Pre-Commit checker로 추가하여 Test를 통과한 코드만 공유하도록 하는 것도 가능한데, 이를 통해 코드의 안정성이 높아지는 효과도 기대할 수 있습니다. 특히 이러한 장점들은 코드가 복잡하고 방대하거나 배포를 염두해두고 있는 상황이라면 더욱 명확해집니다.

물론 이렇게 Test를 자동화하는 것에는 단점도 있는데, 대표적으로는 Test Code를 작성하는 것 자체가 부담스럽다는 것입니다. Test Code까지 생각하게 되면 사실상 작성하는 코드의 양이 2배가 되고, 그에 따라 체감하는 개발 속도 또한 느려져 답답하게 느껴지기도 합니다. 이러한 점 때문에 어느 정도 수준까지 Test를 진행할 것인가에 대해서는 개발자들 사이에서도 갑론을박이 있고, 진행하고 있는 프로젝트의 요구사항, 작업 크기, 참여하는 개발자의 특성 등을 고려하여 적절한 수준으로 도입할 필요가 있습니다.

다만 개인적으로는 Test Code를 작성하는 시간만큼 그에 따른 효율적인 부분들이 있으므로 이로 인해 개발 속도가 크게 느려지지는 않는다고 생각합니다. 오히려 버그를 찾기 위해 어느 부분이 문제인지 print 찍어가며 소모할 시간과 받게 될 스트레스를 줄여주는 등의 긍정적인 경험이 더욱 많았던 것 같습니다. 이러한 점 때문에 최근에는 개인 프로젝트를 진행할 때에도 적극적으로 활용하고 있습니다.

## Software Testing in Machine Learning

앞서 언급한 Test 자동화의 장점들은 사실 프로그래밍과 관련된 영역이라면 분야에 구애받지 않고 누릴 수 있는 장점들입니다. 머신러닝 분야 또한 프로그래밍을 한다는 점에서 Test Code를 도입하여 얻을 수 있는 이점들이 많은데, 구체적으로 회사에서 강화학습 프로젝트를 진행하며 느낀 Test Code 작성의 장점으로는 다음과 같은 것들이 있습니다.

- 디버깅이 쉽고 빨라진다. 따라서 개발 프로세스 자체가 빨라진다.
- 리펙토링이 쉬워진다. 따라서 코드의 readability를 높이는 데에 도움이 된다.
- 다른 사람의 코드를 이해하기 쉬워진다. 따라서 팀 내 커뮤니케이션 비용이 줄어든다.

### 1. Test Code makes debugging easier

머신러닝에서 Test Code가 중요하다고 생각하는 첫 번째 이유는 학습이 잘 되지 않았을 때 Search Space를 줄여주기 때문입니다. 프로그래밍을 하다보면 예상치 못한 반작용(Side Effect)이 종종 발생합니다. 예를 들어 아래 그림에서와 같이 `main` 브랜치에서 새로운 브랜치 `feature/refactor_step`를 따서 작업을 진행하는 상황을 가정해보려 합니다. 이때 Commit `C5`까지 작성을 완료한 후에 무언가 잘못되었다는 것을 알았다면 이미 늦었다고 할 수 있습니다. `C2`에서 문제가 발생하였음에도 이를 찾기 위해서는 브랜치에서 작업한 모든 내용들을 확인해야만 합니다.

<img src="{{site.image_url}}/development/testcode_search_space.png" style="width:48em; display: block; margin: 0em auto;">

반면 Test가 자동화되어 있고, 매 Commit 마다 기능적으로 달라진 점이 없는지 모두 검증했다면 `C2`가 생성되기 이전에 어떤 부분에 문제가 있는지 바로 알 수 있었을 것입니다. 이러한 점은 Pre-Commit Checker로 Test를 등록했을 때의 장점이기도 합니다.

머신러닝의 특성상 학습 성능이 기대 이하인 경우가 자주 발생하고, 학습이 되지 않는 이유를 찾기 위해 디버깅을 실시하는 경우가 잦습니다. 이러한 상황에서도 Test Code를 잘 작성해두면 Search Space가 줄어들게 됩니다. 예를 들어 강화학습 프로젝트를 진행한다고 하면 가장 먼저 하는 일 중 하나가 Agent와 상호작용하는 환경(시뮬레이터)를 만들거나, 이를 Agent와 연결하는 작업이 될 것입니다. 이때 Test Code를 잘 작성해두고 이에 따라 검증을 완료한 상태라면 모델이 학습되지 않더라도 환경과 관련된 부분에 대해선 디버깅을 할 필요가 없어집니다.

### 2. Test Code makes refactoring easier

Refactoring은 기존 코드의 기능은 그대로 유지하면서 코드의 Maintinability 혹은 Readability를 높이는 작업이라고 할 수 있습니다. 하지만 코드만 두고 본다면 변경이 생긴 것이므로 Refactoring를 진행한 후에 제대로 동작하는지 검증해야하는데, 이것이 쉽지만은 않습니다. Test Code를 미리 만들어두었다면 코드 변경 후에도 검증하는 것이 매우 쉬워지고, 기능적으로 변경이 있는지 빠르게 확인하는 것이 가능해집니다. 결과적으로 각각의 기능을 검증하는 Test Code를 잘 작성해두면 Refactoring에 소요되는 비용이 줄어들게 되어 코드의 질 또한 높아지게 됩니다.

### 3. Test Code reduces communication costs

Test Code를 잘 작성해두게 되면 다른 사람이 내가 작성한 코드를 이해하는 데에도 많은 도움이 되는 것 같습니다. Software Testing의 한 종류인 Unit Test의 목적은 특정 부분이 의도대로 동작하는지 확인하는 것인데, 이를 뒤집어보면 Test Case만 잘 확인하면 어떠한 의도를 가지고 작성된 코드인지, 나아가 전체 소스 코드에서 어떠한 기능을 담당하고 있는지 알 수 있음을 알 수 있습니다. 이러한 점에서 잘 쓰인 Test Code는 코드의 모호성을 줄여주고, 코드와 관련된 팀원들 간의 커뮤니케이션 비용을 줄여주는 효과를 만들어 냅니다.

## Notes for writing Test Code

회사에서 Test Code를 작성하며 받은 동료들의 피드백과 개인적인 생각들을 정리해보니 Test Code를 작성할 때 다음과 같은 요소들을 고려하면 좋을 것 같습니다. 여기서 언급하는 요소들은 모두 Python으로 강화학습 프로젝트를 진행하면서 느낀 점이라는 것을 밝힙니다.

##### 1. Test Code가 Test 대상에 종속되면 안 된다

쉽게 말해 Test Case를 작성할 때에는 소스 코드를 최대한 사용하지 않아야 합니다. 예를 들면 어떤 Class의 Method들을 검증한다고 하면 해당 Class의 Attribute와 Method를 사용하는 것을 지양하는 것입니다. 다만 이렇게 Test 대상과 분리를 하려하면 자연스럽게 Test Code를 위한 하드 코딩이 늘어나기도 합니다. 이것이 작성 시에는 다소 부담이 되고, 거부감이 들기도 합니다. 하지만 이상적인 데이터 형태를 명시적으로 확인할 수 있어 이후 코드를 이해하는 데에 도움이 되기도 합니다.

#### 2. 어떤 부분이 문제인지 빠르게 확인할 수 있어야 한다

Test의 목적은 현재 코드에서 작성자가 원하는 대로 동작하지 않는 부분을 찾아내는 것입니다. 따라서 Test 결과가 나왔을 때 어떤 부분이 문제인지 빠르고 정확하게 확인할 수 있도록 해야 합니다. 이와 관련해선 구체적으로 다음과 같은 점들을 고려하는 것이 도움이 되었습니다.

- Test Case는 최대한 잘게 쪼개어주는 것이 좋다.
- Test Case의 이름에서 검증 대상이 드러나야 한다.
- Test Case의 특성에 따라 그룹화하는 것이 좋다.

##### 3. 함수를 검증한다면 입출력 검증은 필수다

함수는 어떤 입력이 주어졌을 때 원하는 출력 값을 반환해야 합니다. 따라서 입출력을 검증하는 것이 필수적이라 할 수 있는데, 이와 관련하여 다음과 같은 점들을 확인해보는 것이 좋습니다.

- 입력의 범위, 입력의 타입이 정해져 있다면 이에 대한 check가 필요하다.
- 가능한 Edge Case를 list-up 하고 모두 검증할 수 있도록 한다.
- Method의 경우에는 중간에 Attribute를 변경하기도 하는데, 이에 대해서도 출력과 마찬가지로 검증한다.

Edge Case를 선정할 때에는 가능한 State를 모두 나열해보는 것이 중요합니다. 예를 들어 다음과 같이 하나의 Method 내에 if 문에 총 3개 있다면 8개의 State가 존재하므로, 모든 State를 검증하기 위해서는 최소 8개의 Edge Case가 필요합니다.

```python
def it_has_eight_state(inp1: int, inp2: int, inp3: int) -> int:
    number = 0
    if inp1 > 0:
        number += 1
    if inp2 > 0:
        number += 1
    if inp3 > 0:
        number += 1
    return number
```

| inp1  | inp2 | inp3 | return |
|:---:|:---:|:---:|:---:|
| 0  | 0  | 0  | 0  |
| 0  | 0  | 1  | 1  |
| 0  | 1  | 0  | 1  |
| 0  | 1  | 1  | 2  |
| 1  | 0  | 0  | 1  |
| 1  | 0  | 1  | 2  |
| 1  | 1  | 0  | 2  |
| 1  | 1  | 1  | 3  |

##### 4. Test Case마다 그 목적이 무엇인지 Docstring을 명확히 작성해야 한다

Test Case마다 Docstring으로 무엇을 검증하기 위한 것이며, 입력과 출력은 어떠해야 한다는 것을 명시해주어야 합니다. 실제 프로젝트에서 사용한 예시는 다음과 같습니다.

```python
def test_reward_and_done_with_step_iteration(self):
    """Check return(reward and done value) of the step function is correct.
    
    CheckList:
        - The reward is always zero except the last step.
        - The done is always False except the last step.

    Notes:
        - The 72th step is the last step.
    """
```

##### 5. Test Case만 보고도 어떻게 동작하는지 이해할 수 있어야 한다

Test Case의 첫 번째 목적이 작성한 코드를 검증하는 것이라면, 두 번째 목적은 작성한 코드를 설명하는 것이라고 생각합니다. 따라서 Test Case를 작성한다면 처음 코드를 보는 사람도 쉽게 이해할 수 있도록 깔끔하게 작성할 필요가 있습니다.

##### 6. Test Code도 유지 보수의 대상이다

Test Code 또한 유지 보수의 대상이며, 새로운 기능을 추가하거나 기존 기능에 변경이 생기면 그에 맞추어 업데이트 되어야 합니다. 따라서 Test Code 또한 효율성을 고려하지 않을 수 없으며, 경우에 따라서는 위에서 언급한 요소들에 대해 타협을 해야 하기도 합니다. 또한 이러한 점에서 'Test Covarage를 80% 이상으로 끌어올려야 한다'와 같는 규칙들은 지양하는 것이 좋다고 생각합니다.
