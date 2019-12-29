---
layout: post
title: AUPR) The relationship between precision-recall and ROC curves
---

# 논문 제목 : The relationship between precision-recall and ROC curves

- Jesse Davis 등
- 2006
- <https://www.biostat.wisc.edu/~page/rocpr.pdf>
- 2019.10.07 정리

## 세 줄 요약

1. 알고리즘의 성능 평가에 널리 사용되는 ROC curve의 경우 클래스 불균형 데이터셋에서는 성능을 과대평가할 가능성이 있으며, 이에 대한 대안으로 Recall-Precision curve를 제시한다.
2. PR space 상에서 linear interpolation을 이용해 PR curve를 그리는 방법은 좋지 않으며, convex hull ROC curve를 이용해 achievable PR curve를 그릴 수 있다.
3. AUROC 상에서 가장 좋은 알고리즘이라고 하여 AUPR 상에서도 가장 좋은 알고리즘이라고는 할 수 없다.

## 내용 정리

### 알고리즘의 성능 평가에 사용되는 metric

- 이진분류문제에서는 레이블을 positive, negative로 나누어 보게 되고, 각각에 대한 예측의 결과를 다음과 같이 표현할 수 있다.

|prediction|actual positive|actual negative|
|------|:---:|:---:|
|predicted positive|*TP(True positive)*|*FP(False positive)*|
|predicted negative|*FN(False negative)*|*TN(True negative)*|

- Recall : TP / TP+FN
- Precision : TP / TP + FP
- TPR(true positive rate) : TP / TP + FN
- FPR(false positive rate) : FP / FP + TN
- Recall 과 TPR 은 동일하다.
- ROC curve와 PR curve는 알고리즘의 성능을 평가하는 데에 사용되는 metric 이다.
  - ROC curve는 x축으로 FPR, y축으로는 TPR을 사용한다.
  - PR curve는 x축으로 Recall, y축으로는 Precision을 사용한다.

### ROC curve

- ROC curves란 Reveiver Operator Charachteristic curves를 말한다.
- Provest 등은 1998년 논문에서 accuracy 만을 가지고 모델을 평가하는 것은 부족하다고 주장하며, ROC curve 를 대안으로 제시했다.
- ROC curves는 정확하게 분류된 positive example의 수가 잘못 분류된 negative example의 수에 따라 어떻게 변화하는지를 보여준다.
  - "ROC curves show how the number of correctly classified positive examples varies with the number of incorrectly classified negative example"
- 하지만 ROC curves는 클래스 분포 불균형 데이터셋(large skew in the class distribution)에서는 알고리즘의 성능을 과도하게 낙관적으로 평가하는 경향을 보이는 문제점이 있다.
- 이러한 문제를 해결하기 위해 Drummon 등이 2000년에 제시한 Cost curves라는 것이 있다.
- Cost curve를 논외로 하며, 논문에서는 Precision-Recall Curve(PR curves)를 제안하고 있다.

### PR curve

- 클래스 분포 불균형 데이터 셋에서 ROC curve의 대안으로 제시되었다.
- 논문의 figure 1로 제시된 그림에서는 클래스 분포 불균형 데이터 셋에 대한 두 개의 알고리즘에 대한 ROC curve와 PR curve를 비교하고 있다.
  - ROC curve의 경우 좌측 상단(upper-left-hand)에 curve가 가까워질수록 성능이 높은 알고리즘이 된다. 그런데 커브 만으로는 알고리즘 1과 알고리즘 2 중 무엇이 성능이 더 좋은지 알기 힘들다.
  - 반면 우측 상단(upper-right-hand)에 가까울수록 성능이 높은 PR curve 그림에서는 알고리즘 2가 알고리즘 1보다 더 좋다는 것을 쉽게 확인할 수 있다
  - 결론적으로 클래스 불균형 분포 데이터셋 환경에서는 PR curve가 ROC curve보다 성능 확인에 유리하다.
- PR curve는 ROC curve와 달리 precision을 본다는 점에서 이러한 차이의 이유를 유추할 수 있다.
  - false positive를 true negative가 아닌 true positive와 비교하기 때문에 많은 수의 negative sample이 주는 영향을 잡아낼 수 있다는 것이다.

### ROC curve 와 PR curve의 관계

#### 1. ROC space 상의 curve와 PR space 상의 curve는 서로 1:1 대응이다

- recall이 0이 아닐 때에만 성립한다.
- 하나의 confusion matrix로는 하나의 ROC curve만 그릴 수 있다.
  - ROC curve의 경우 confusion matrix 상의 TP, TN, FP, FN 모두 사용한다.
- PR curve 또한 하나의 confusion matrix로 하나의 PR curve만 그릴 수 있다.
  - PR curve의 경우 TN을 사용하지 않지만, recall이 0이 아니라면 TN을 유추할 수 있기 때문이다.
  - "TN is uniquely determined. If recall = 0, we are unable to recover FP, and thus cannot find a unique confusion matrix"

#### 2. ROC space 상에서 우세한 curve는 PR space 상에서도 우세한 curve를 가진다

#### 3. PR space 상에서 우세한 curve는 ROC space 상에서도 우세한 curve를 가진다

- 2와 3은 귀납법을 통해 증명된다.
- 2와 3이 모두 만족하기 때문에 필요충분관계가 성립한다.

### ROC curve를 그리는 방법

- ROC space에 curve를 그릴 때에 생각해야 할 부분 중 하나는 space 상의 점들을 잇는 방법이다.
- convue hull은 ROC 점들을 이용해 curve를 그리는 방법들 중 가장 좋은 방법 중 하나이다.
  - "The convex hulll in ROC space is the best legal curve that can be constructed from a set of given ROC points"
- convex hull에 관한 내용은 Cormen 등의 1990년 논문에 자세히 나와있다.

#### convex hull의 조건

  1. 두 점간을 이을 때에는 linear interpolation을 사용한다.
  2. 최종 커브 상에는 어떠한 점도 없다.
  3. 어떤 점들의 조합으로 커브를 그리더라도 convex hull 아래에 위치해야 한다.

### PR curve를 그리는 방법

- PR space에서 interpolation이 ROC space보다 복잡하다.
  - 단순히 linear interpolation을 하게 되면 성능을 과도하게 낙관적으로 평가하게 될 위험이 있다.
  - "linear interpolation is a mistake that yields an overly-optimistic estimate of performance"
- 직접적으로 linear interpolation을 하는 것보다는 ROC convex hull을 PR space로 변환하는 것도 하나의 방법이 된다. 본 논문에서는 이를 "acheivable PR curve"라고 부른다.
- 구체적으로는 다음과 같이 PR space 상에 PR curve를 그릴 수 있다.

1. ROC space에 convex hull을 그린다.
2. 이렇게 그려진 convex hull 상에 위치하고 있는 점들을 PR space에 옮겨 찍는다(1:1 대응이기 때문에 가능하다).
3. 이렇게 구해진 점들을 이용해 PR space 상에서 interpolation을 실시한다.

#### ROC curve 상에서 최고의 algorithm이 PR curve 상에서도 최적의 algorithm이라고 할 수 있을까

- 결론적으로 말하자면 아니다. 즉 AUC-ROC 상에서 최적의 알고리즘이라 하더라도 AUC-PR 상에서는 그렇지 않을 수 있다
- 이러한 차이는 PR space에서는 lower Recall과 higher Precision으로 구해지기 때문이다.
- 즉 ROC curve 상에서 최적이라고 하여 PR curve를 그릴 필요가 없어지는 것은 아니다. 각각에 대해 PR curve를 그려보고 AUPR을 비교해 볼 필요가 있다.
