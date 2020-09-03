---
layout: post
title: 1. Meta Learning & Multi-task Learning
category_num : 1
---

# Meta Learning & Multi-task Learning

- Chelsea Finn의 Stanford [CS330 Deep Multi-Task and Meta Learning](<https://cs330.stanford.edu/>)을 참고하여 작성했습니다.
- 포스팅에 사용된 이미지 또한 동일한 강의의 PPT를 활용했습니다.
- Update at: 2020.08.30

## 0. Introduction: Problem of Deep Learning

딥러닝은 일반적으로 학습을 위해 큰 데이터 셋이 필요하다. 분류 알고리즘의 성능을 확인하기 위해 가장 일반적으로 사용되는 Dataset인 MNIST도 6만 장으로 이뤄져있다. 하지만 현실에서는 방대한 데이터를 확보하는 것이 어려운 경우가 많고, 이러한 점은 딥러닝을 적용하는데 있어 걸림돌이 되어왔다. Multi Task Learning과 Meta Learning은 개별 task를 처음부터(from scratch) 배우는 것은 비효율적이라 지적하며, Data의 수가 적은 경우에 대해서도 효과적으로 문제를 해결할 수 있는 방법을 찾아내는 것에 관심을 가지는 분야라고 할 수 있다.


### Shared Structure

이러한 문제와 관련하여 Multi-Task Learning과 Meta Learning에서는 많은 문제들이 유사한 특성을 공유한다는 점에 집중한다. 쉽게 생각해 물병을 따는 로봇 알고리즘을 개발했다면 약간의 튜닝을 통해 딸기잼 뚜껑을 따거나 참치캔을 여는 알고리즘을 빠르게 학습시킬 수 있을 것이라 보는 것이다. 이를 보다 딱딱하게 표현하면 Task 간의 공통된 구조(Shared Structure)를 찾고 이를 통해 보다 효율적으로 네트워크를 구성하거나 학습하는 문제에 관심을 가진다고 말할 수 있다.

## 1. Multi-Task Learning & Meta Learning

여러 문제들이 가지는 공통 구조에 관심을 가진다는 점에서 두 가지는 비슷하지만 구체적으로는 다음과 같은 차이를 가지고 있다.

- **Multi Task Learning**: 복수의 task를 동시에 효율적으로 배우는 것에 관심을 가진다.
- **Meta Learning**: 과거의 지식을 토대로 새로운 task를 보다 빠르고 효율적으로 학습하는 방법에 관심을 가진다.

### Task

그렇다면 **Task**란 무엇일까. 예를 들어 이미지를 분류하는 문제가 있다고 하자. Dataset $$D = \{ (x, y)_k \}$$이 있을 것이고 이를 학습하여 정확하게 입력 $$x$$이 들어오면 Label $$y$$를 예측하는 Model $$f_{\theta}(y \lvert x)$$을 만드는 것이 목표가 된다. 이러한 목표를 달성하기 위해 (Typical) loss function $$L(\theta, D) = -E_{(x,y)\backsim D}[\log f_\theta(y \lvert x)]$$도 필요하다. 이때 Task는 다음 세 가지로 정의할 수 있다.

$$
T_i \triangleq \{ p_i(x), p_i(y \lvert x), L_i \}
$$

여기서 $$p_i$$는 Task $$T_i$$의 입력의 분포를, $$p_i(y \lvert x)$$는 입력이 주어졌을 때 레이블의 분포를, 그리고 $$L_i$$는 loss function을 의미한다

## 2. Multi-Task Learning

앞서 언급한 대로 Multi-Task Learning은 여러 개의 Task를 모두 해결할 수 있는 Model을 학습시키는 방법이다. 즉, 여러 개의 Task가 있고 이를 모두 해결할 수 있는 Model을 만들고 싶은 것이다. 이때 각 Task의 Task index를 $$z_i$$라고 한다면 Model은 다음과 같이 정의된다.

$$
f_\theta(y \lvert x, z_i)
$$

그리고 최적화 식은 다음과 같다.

$$
\min_\theta \Sigma_{i=1}^T L_i(\theta, D_i)
$$

### Two Extreme Case

Multi-Task Learning의 문제를 해결하는 방법과 관련하여 가장 먼저 떠올릴 수 있는 방법은 Task의 개수만큼 각각 모델 $$f_{\theta}(y \lvert x)$$을 학습하도록 하는 것이다.

<img src="{{site.image_url}}/study/multi_task_learning_independent_training.png" style="width:34em; display: block; margin: 0px auto;">

위의 경우 각각의 Task는 모델을 거의 공유하지 않는다. 반면 모든 Task들이 전체 모델을 대부분 공유하도록 하는 것도 가능하다. 즉 아래와 같이 $$z_i$$를 네트워크 중간에 concat하여 끼워넣는 방법이다.

<img src="{{site.image_url}}/study/multi_task_learning_concat_z_training.png" style="width:34em; display: block; margin: 0px auto;">

이러한 점에서 생각해 본다면 $$z_i$$에 대한 조건을 설정하는 것은 **모델의 어떤 부분은 Task 공유할지 결정하는 것**과 동일하다고 할 수 있다. 모델의 공유하는 파라미터를 $$\theta_{sh}$$라고 하고, 개별 Task마다 별개로 사용하는 파라미터를 $$\theta_i$$라고 한다면 목적 함수를 다음과 같이 다시 정의할 수 있다.

$$
\min_{\theta_{sh}, \theta_1, ... \theta_{T}} \Sigma_{i=1}^T L_i(\{ \theta_{sh}, \theta_i \}, D_i)
$$

### Optimizing the Multi-Task Laerning Objective

Multi-Task Laerning의 기본적인 학습 과정은 다음과 같다.

1. Task $$\{ T_i \}$$에서 mini-batch $$B \backsim \{ T_i \}$$를 Uniform하게 뽑는다.
2. Task에 대한 mini-batch data $$D_i^b \backsim D_i$$를 뽑는다.
3. mini-batch에 대한 loss $$L(\theta, B) = \Sigma_{T_k \in B} L_k (\theta, D_k^b)$$ 를 구한다.
4. back-propagation: $$\nabla_\theta L$$
5. Optimization

### Challenges of Multi-Task Learning

Multi-Task Learning의 주요 문제로는 **Negative Transfer**와 **Overfitting** 두 가지가 있으며, 이들은 Model을 얼마나 공유하는가와 관련하여 Trade-off 관계를 가진다.

#### Negative Transfer

Negative Transfer란 쉽게 말해 과거에 습득한 지식이 현재의 학습에 악영향을 미치는 것을 말한다. 이를 해결하기 위해서는 Network를 공유하는 정도를 낮추어야 한다.

#### Overfitting

Overfitting 문제는 몇몇 Task에만 너무 집중하여 다른 Task들은 잘 해결하지 못하는 문제를 말한다. 이를 해결하기 위해서는 Network를 공유하는 정도를 높이거나, 모델의 크기를 키워야 한다.

## 3. Meta Learning

Meta Learning는 Supervised Learning을 목적으로 Deep Learning 모델을 학습시키기 위해서는 다수의 labeled data가 필요하지만 현실적으로 어려운 경우가 많다는 문제의식에서 출발한다. 이를 해결하기 위해 Meta Learning은 복수의 Task로 구성되어 있는 Meta Dataset으로부터 Prior Information을 먼저 학습하고 이를 사용하여 새로운 Task를 효율적으로 학습하도록 하는 방법으로 접근한다. 이에 대해 강의에서는 다음과 같이 표현하고 있다.

- Extract prior information from a set of tasks that allows efficient learning of new tasks
- Learning a new task uses this prior and (small) training set to infer most likely posterior parameters

### Find Optimal Meta Parameter

따라서 Meta Learning의 특징을 다음과 같이 두 가지로 정리할 수 있다.

- 여러 Task들에 대해 먼저 학습하고, 이를 통해 획득한 정보를 Prior로 사용한다.
- 획득한 정보를 통해 새로운 Task를 적은 데이터를 가지고도 효율적으로 학습할 수 있도록 한다

이때 '다른 Task들에 대해 먼저 학습'하는 것을 **Meta Training**이라 하고, 이를 위해 사용되는 Dataset을 **Meta Dataset**이라고 한다. 그리고 Parameter 형태로 전달되는 '획득한 정보'를 **Meta Parameter**라고 한다. 수학적으로 표현하면 다음과 같다.

$$
\eqalign{
&\eqalign{
\text{Meta-Learning: } &\arg \max_\phi \log p(\phi \lvert D, D_{\text{meta-train}})\\
&D = \{ (x_1, y_1), ..., (x_k, y_k) \}\\
&D_{\text{meta-train}} = \{ D_1, ..., D_n \}\\
}\\ \\ \\
&\text{Meta-Parameter } \theta: p(\theta \lvert D_{\text{meta-train}})\\
& \eqalign{ \log p(\phi \lvert D, D_{\text{meta-train}}) &= \log \int_\theta p(\phi \lvert D, \theta) p(\theta \lvert D_{\text{meta-train}})d\theta\\
&\approx \log p(\phi \lvert D, \theta^*) + \log p(\theta^* \lvert D_{\text{meta-train}})
} \\ \\ \\
& \therefore \arg \max_\phi \log p(\phi \lvert D, D_{\text{meta-train}}) \approx  \arg \max_\phi \log p(\phi \lvert D, \theta^*) \\
& \quad \text{where } \theta^* = \arg \max_\theta \log p(\theta \lvert D_{\text{meta-train}}) \leftarrow \text{Meta Learning Problem}
}\\
$$

예시를 통해 이해하면 보다 쉽게 이해할 수 있는데, Meta Learning을 통해 풀고자 하는 문제를 Dataset $$D$$에 들어있는 사진만을 사용하여 Test Data를 분류하는 것이라고 하자. 문제는 Dataset $$D$$에 포함된 사진의 숫자가 매우 적어 일반적인 방법으로는 학습이 어렵다는 것이다. 이를 해결하기 위해 Meta Learning은 풀고자 하는 문제와 유사한 Task, 즉 데이터는 다르지만 적은 데이터로 구성된 $$D_{\text{meta-train}}$$에 대해 분류하는 방법을 먼저 학습하고 그렇게 얻어낸 Parameter, $$\theta^*$$를 사용한다. 이를 도식화하면 다음과 같다.

<img src="{{site.image_url}}/study/quick_example_of_meta_learning.png" style="width:30em; display: block; margin: 0px auto;">

전체적으로 구조를 보게 되면 $$D_{\text{meta-train}}$$으로 미리 학습된 모델 $$\theta^*$$에서 $$D$$를 입력으로 받아들이고 여기서 얻은 정보를 활용하여 Test Input을 정확하게 분류할 수 있도록 $$D$$를 분류하는 데에 특화된 $$\phi$$를 학습하도록 하게 된다. 이때 미리 $$\theta$$를 학습하는 과정을 **Meta Learning Process**, 각 Task에 맞게 $$\phi$$를 학습하도록 하는 것을 **Adaptation Process**라고 한다.

### Test and Train Conditions Must Match

- “our training procedure is based on a simple machine learning principle: test and train conditions must match”(Vinyals et al)

한 가지 중요한 것은 Machine Learning에서 Training을 진행할 때와 Test를 진행할 때의 조건이 동일해야 한다는 점이다. 따라서 Test Time을 나타내는 위의 그림과 형식적으로 동일하게 Train Time에 학습이 이뤄져야 한다. 즉 Train Dataset 또한 입력으로 들어가는 $$D_i$$와 이를 통해 추출한 정보를 사용하여 분류하게될 Test Data $$D_i^{ts}$$로 나누어야 한다. Train Time은 다음과 같이 도식화할 수 있다.

<img src="{{site.image_url}}/study/quick_example_of_meta_learning_train_time.png" style="width:24em; display: block; margin: 0px auto;">

이때 $$D_{\text{meta-train}}$$의 Dataset에서 학습에 사용되는 Data $$D_i$$를 **Support Set**이라 하고, 분류의 대상이 되는 Data $$D_i^{ts}$$를 **Batch Set**이라고 한다.

문제 정의와 관련된 용어로 $$n$$ way $$m$$ shot과 같은 표현들이 자주 나오는데, 여기서 $$n$$ way는 $$D_{\text{meta-train}}$$에 포함된 Dataset의 개수(Task)를 의미한다고 할 수 있다. 그리고 $$m$$ shot은 개별 Meta Dataset에서 각 Class에 포함되어 있는 Data의 갯수를 뜻한다.
