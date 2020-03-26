---
layout: post
title: mlflow Introduction
category_num : 5
---

# MLFLOW INTRODUCTION

- update date : 2020.02.15

- [mlflow quick start](<https://mlflow.org/docs/latest/quickstart.html>)
- [mlflow python API](<https://mlflow.org/docs/latest/python_api/mlflow.html#>)
- [mlflow command line interface](<https://mlflow.org/docs/latest/cli.html#cli>)

- ubuntu 18.04.2 LTS, mlflow version 1.6.0을 기준으로 작성했습니다.

## 0. Installation

mlflow는 virtualenv 환경에서 설치하는 것을 권장하고 있습니다.

```
$ pip install mlflow
```

numpy, pandas와 같은 기본적인 package가 함께 설치되지만 torch는 별도로 설치해주어야 합니다.

```
$ pip install torch torchvision
```

## 1. Tracking

mlflow의 주요 기능 중 하나는 실험 결과를 관리하고 GUI를 통해 직관적인 실험 결과 확인 및 결과 간의 비교가 가능하다는 것입니다. 실험 결과를 저장/관리/확인하는 과정을 mlflow에서는 **tracking**이라고 합니다.

tracking은 다음과 같은 과정을 거쳐 이뤄집니다.

1) 저장 데이터 설정하기
2) 학습 진행하기
3) GUI server 열기

### 0) run과 experiment

tracking과 관련하여 가장 기본적인 개념으로 **run**과 **experiment**가 있습니다. 두 가지 모두 모델을 학습시키고 결과를 저장하는 단위가 되며, mlflow를 사용하는데 있어 필수적인 구성요소입니다.

#### runs

mlflow를 적용하여 모델 학습을 진행하면 그와 관련된 모든 정보들은 `run`이라는 개별 단위로 저장 관리됩니다. 구체적으로 `run`은 Source, Versions, Start&end time, Parameters, Metrics, Tags, Artifacts 등 7개로 구성되며, 이 중 Source, Versions, Start&end time는 개별 run의 메타 정보를 담고 있는 meta.yaml에 저장되고 다른 요소들은 개별 디렉토리로 관리됩니다.

- Parameters: 학습 시 사용된 하이퍼 파라미터
- Metrics: train_loss, accuracy 등 실험 결과 및 과정
- Artifacts: 학습 모델, 데이터, 기록 이미지 등 학습 결과로 생성된 다양한 형식의 데이터
  - 모델을 저장하게 되면 `artifacts` 디렉토리 내에 저장

##### CLI

- runs 확인

    ```
    $ mlflow runs list --experiment-id <experiment_id:required>
    ```

- 개별 run 확인

    ```
    $ mlflow runs describe --run-id <run_id:required>
    ```

#### experiement

experiment는 run의 상위 개념으로, mlflow의 모든 run은 experiment에 포함됩니다. experiment는 시각화 및 분석의 단위로 기능합니다.

experiment를 별도로 생성/지정하지 않으면 default experiment인 `0`으로 저장됩니다. experiment 또한 메타 정보를 담고 있는 meta.yaml을 가지고 있습니다.

##### CLI

- experiment 생성

    ```
    $ mlflow experiments create -n <experiment_name>
    ```

- experiment 확인

    ```
    $ mlflow experiments list
    ```

##### Python API

- experiment 생성

    ```
    mlflow.set_experiment(experiment_name)
    ```

### 1) 저장 데이터 설정하기

mlflow에서 저장 데이터를 설정하는 과정은 tensorboard와 매우 유사한데, 실행하고자 하는 코드 사이사이에 tracking method를 추가하는 방식으로 이뤄집니다.

예제 코드는 다음과 같습니다.

```
import mlflow

# run 열기
mlflow.start_run()

# 학습 과정에서 데이터 저장하기
mlflow.log_param("batch_size", 64)
mlflow.log_metric("train_loss", loss, step)

# run 닫기
mlflow.end_run()
```

#### tracking 범위 설정

우선 첫 번째로 기록의 단위가 되는 `run`을 열어주어야 합니다.

```
mlflow.start_run()
...
mlflow.end_run()
```

위 두 가지 method로 열고 닫을 수 있는데, 쉽게 `with` 키워드로 파이썬 스코프 설정을 이용하는 것을 권장하고 있습니다. 

```
with mlflow.start_run() as f:
    ...
```

`mlflow.start_run()`은 필수 파라미터를 가지지 않습니다. 하지만 특정 experiment로 관리하고 싶은 경우에는 아래와 같이 특정 id 값을 함께 전달하면 됩니다.

- experiment 설정 : `mlflow.start_run(experiment_id = <experiment id>)`

    ```
    mlflow.start_run(experiment_id = 1)`
    ```

    - 이때 experiment_id로 매칭되는 experiment가 없는 경우에는 error가 발생합니다 따라서 experiment를 먼저 생성해주고 해당 id를 부여해야 합니다.

#### 저장 데이터 지정

tensorflow, keras 등에서는 automatic logging을 실험적으로 지원하고 있지만 torch는 빠져있습니다. 따라서 저장하고 싶은 데이터를 개별적으로 설정해주어야 합니다.

- 하이퍼 파라미터 저장 : **mlflow.log_param(key, value)**

    ```
    mlflow.log_param("batch_size", 64)
    ```

- metric 저장 : **mlflow.log_metric(key, value, step=None)**

    ```
    mlflow.log_metric("train_loss", loss, step)
    ```

- model 저장(torch) : **mlflow.pytorch.log_model(model, model_name)**

    ```
    mlflow.pytorch.log_model(model, "test_model")
    ```

    첫 번째 파라미터로 지정된 torch model을 두 번째 파라미터로 전달된 이름으로 저장합니다. 자세한 사항은 3. Model에 나와 있습니다.


### 2) 학습 진행하기

저장 데이터를 지정한 후 학습을 완료하게 되면 `mlruns` 디렉토리 내부에 uuid 값으로 개별 `run` 디렉토리가 생성되어 관련 데이터가 내부에 저장됩니다.

이와 관련하여 mlflow의 기본 디렉토리 구조는 다음과 같습니다.

```
mlruns
|- .trash
|- 0   // experiment id
    |- 8a3c68a88ed50665c87****
    |- b0ffc465381dbfb77995d231d****   // 개별 run의 uuid
      |- artifacts  // 모델과 관련된 내용들이 저장되는 디렉토리
      |- metrics   // 로그 데이터가 저장되는 디렉토리
      |- params   // 하이퍼 파라미터가 저장되는 디렉토리
      |- tags
      |- meta.yaml   // run의 메타 정보가 저장되는 파일
    |- fd4bc0fba6834414
```

### 3) GUI server 열기

`run`이 저장된 경우 아래 명령어를 활용해 GUI SERVER를 열어 확인할 수 있습니다.

```
$ mlflow ui
```

또는

```
$ mlflow server
```

두 가지 모두 가능합니다. 기본 port는 5000이고 `-p` 옵션을 통해 변경 가능합니다.

#### connection denied 또는 404 error

mlflow GUI server와 관련하여 connection error가 자주 발생했습니다. 이는 다음과 같은 경우 발생합니다.

- **port가 이미 사용중인 경우**

    mlflow gui server를 종료하여도 process는 살아있는 경우가 자주 있었습니다. `$ps ax | grep 5000`으로 찾고, 포트를 점유하고 있는 프로세스를 `$ kill <id>`로 죽이면 됩니다.

- **mlruns를 찾을 수 없는 경우 또는 잘못 찾은 경우**

    server를 열 때 mlruns가 저장되어 있는 디렉토리에서 실행해야 합니다.

## 2. Projects

mlflow에서 **project**란 이미 학습된 모델을 연속하여 학습하거나 동일한 코드를 재사용하여 학습하고자 할 때 기본 단위가 되는 것을 말합니다.

### MLproject File

project는 `MLproject`라는 이름으로 관리되며, name, entry points, environmnet로 구성됩니다.

```
<MLproject File>

name: my_project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      batch-size: {type: int, default: 64}
      test-batch-size: {type: int, default: 1000}
      epochs: {type: int, default: 1}
      lr: {type: float, default: 0.01}
      momentum: {type: float, default: 0.5}
      enable-cuda: {type: string, default: 'True'}
      seed: {type: int, default: 5}
      log-interval: {type: int, default: 100}
    command: |
          python mnist_torch.py \
            --batch-size {batch-size} \
            --test-batch-size {test-batch-size} \
            --epochs {epochs} \
            --lr {lr} \
            --momentum {momentum} \
            --enable-cuda {enable-cuda} \
            --seed {seed} \
            --log-interval {log-interval}
```

#### 1) name

project의 이름을 설정하는 프로퍼티입니다.

#### 2) environment

위의 예시에서는 conda를 환경으로 사용하며, 이 경우 conda.yaml이 동일한 디렉토리 내에 위치해 있어야 합니다. 상대경로로 환경설정파일의 위치를 별도로 지정하는 것 또한 가능합니다.

mlflow에서는 conda environment, docker container environment 두 가지가 가능하며, 별도의 환경 설정 없이 로컬 환경을 사용하는 것 또한 가능합니다.

##### conda environment

위의 예시와 같이 `conda_env` 키 값을 이용하여 사용하고자 하는 conda.yaml 파일을 지정해주어야 합니다. 양식만 맞추면 아래 예시와 같이 conda.yaml가 아닌 다른 이름으로도 가능합니다.

```
conda_env: files/config/conda_environment.yaml
```

conda environment file과 관련해서는 다음 [링크](<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually>)를 참조하시기 바랍니다.

##### docker container environment

mlflow에서는 지정된 docker image에서 실험을 진행하는 것 또한 지원합니다. 이와 관련하여 [공식 홈페이지](<https://mlflow.org/docs/latest/projects.html>)에서는 다음 세 가지 경우를 제시하고 있습니다.

- Image without a registry path

    아래와 같이 구체적으로 docker image의 위치가 지정되지 않은 경우에는 MLproject 파일이 있는 위치를 우선 탐색합니다. 이때 동일한 이름의 docker image가 발견되지 않으면 Dockerhub를 탐색하고, 발견되는 경우에는 pull하여 사용합니다.

    ```
    docker_env:
        image: mlflow-docker-example-environment
    ```

- Mounting volumes and specifying environment variables

    아래와 같이 docker image 관련 인자를 전달하여 사용하는 것 또한 가능합니다.

    ```
    docker_env:
        image: mlflow-docker-example-environment
        volumes: ["/local/path:/container/mount/path"]
        environment: [["NEW_ENV_VAR", "new_var_value"], "VAR_TO_COPY_FROM_HOST_ENVIRONMENT"]
    ```

- Image in a remote registry

    리모트에 저장된 docker image를 사용하고 싶은 경우에는 다음과 같이 할 수 있습니다.
 
    ```
    docker_env:
        image: 012345678910.dkr.ecr.us-west-2.amazonaws.com/mlflow-docker-example-environment:7.0
    ```

##### local environment

환경설정과 관련하여 아무것도 설정하지 않으면 됩니다. 즉 위의 예시에서 `conda_env: conda.yaml` 줄을 삭제하면 됩니다. 하지만 이 경우 project를 실행할 때 `--no-conda` 옵션을 추가해 실행해야 합니다.

#### 3) Entry Points

entry point는 진입점이라는 표현대로 MLproject가 실행되기 시작할 때 사용되는 요소들을 관리합니다. 위의 예시와 같이 parameter와 실행되는 command line으로 구성됩니다.

##### parameters:

```
entry_points:
  main:
    parameters:
        batch-size: {type: int, default: 64}
        epochs: {type: int, default: 1}
        lr: {type: float, default: 0.01}
        ...
```

위와 같이 실험에 사용하는 파라미터 값을 설정할 수 있습니다.

##### command:

```
entry_points:
  main:
    ...
    command: |
          python mnist_torch.py \
            --batch-size {batch-size} \
            --epochs {epochs} \
            --lr {lr}
            ...
```

실행 파일(위의 경우 mnist_torch.py)을 command line에서 실행하고자 할 때 사용하는 명령어를 그대로 작성합니다.

##### multi entry points

아래와 같이 복수의 entry point를 설정하는 것 또한 가능합니다. default entry point는 `main` 입니다.

```
entry_points:
  main:   // Entry point 1
    parameters:
      data_file: path
      regularization: {type: float, default: 0.1}
    command: "python train.py -r {regularization} {data_file}"

  validate:    // Entry point 2
    parameters:
      data_file: path
    command: "python validate.py {data_file}"
```

### Running Projects

mlflow에서 설정된 MLproject를 실행하는 방법은 commnad line을 이용하는 방법과 python 코드를 이용하는 방법 두 가지가 있습니다.

##### CLI

command line에서는 기본적으로 mlflow run 명령어를 사용합니다.

```
$ mlflow run [mlflow project dir(uri)] [options]
```

이때 중요한 것은 MLproject 파일이 아닌 MLproject 파일과 conda, docker image 등 환경 및 실행 스크립트 파일을 포함하고 있는 디렉토리 경로를 전달해야 한다는 것입니다.

```
$ mlflow run ./myproject
```

깃허브에 올라가 있는 소스코드를 곧바로 이용하는 것도 가능합니다.

```
$ mlflow run git@github.com:<git address>.git [options]
```

옵션으로는 다음과 같은 것들이 가능합니다. 구체적인 옵션은 `$ mlflow run --help`로도 확인할 수 있습니다.

- parameter

    MLproject 내에 정의된 parameter 값을 변경할 수 있습니다.

    ```
    -P, --param-list NAME=VALUE
    ```

- entry point

    MLproject 내에 정의된 특정 엔트리 포인트를 지정할 수 있으며, 기본값은 `main` 입니다.

    ```
    -e, --entry-point [NAME]
    ```

- experiment name

    어떤 experiment로 관리할 것인지 설정할 수 있습니다.

    ```
    --experiment-name [TEXT]
    ```

    동일한 experiment name이 있는 경우에는 해당 experiment로 저장/관리되지만, 없는 경우에는 새로운 experiment를 생성합니다.

    experiment name과 관련해 유의할 점이 있다면 파이썬 코드의 `mlflow.start_run()`에서 별도의 experiment id를 설정하면 해당 옵션이 반영되지 않습니다.

- experiment id

    experiment name과 동일하게 사용하고자 하는 experiment를 설정할 수 있습니다.

    ```
    --experiment-id [TEXT]
    ```
    
    하지만 이 경우 없는 experiment를 입력하면 새로 생성되지 않고, 에러를 반환합니다/

- run id

    사용하고자 하는 run id를 설정할 수 있게 해줍니다.

    ```
    --run-id RUN_ID
    ```

- no conda

    새로운 conda environment를 생성하지 않고 system environmnet를 곧바로 사용할 수 있도록 해줍니다.

    ```
    --no-conda
    ```

    아래와 같은 에러가 발생하면 `--no-conda` 옵션을 추가해주거나 conda와 관련된 설정을 수정해주어야 합니다

    ```
    ERROR mlflow.cli: === Could not find Conda executable at conda.
    ```

##### python

파이썬에서는 아래와 같이 projects.run 함수를 사용합니다.

```
mlflow.projects.run()
```

구체적인 파라미터의 사용방법은 위의 CLI와 유사합니다.

```
mlflow.projects.run(
    uri, 
    entry_point='main', 
    parameters=None, 
    experiment_name=None, 
    experiment_id=None, 
    use_conda=True, 
    run_id=None
    ...
    )
```

## 3. Model

### 모델 저장하기

mlflow를 통해 학습된 모델을 저장할 수 있습니다. 특정 시점의 학습된 torch 모델을 저장하기 위해서는 아래 코드를 사용합니다.

```
mlflow.pytorch.log_model(
    pytorch_model, 
    artifact_path, 
    conda_env=None, 
    code_paths=None, 
    pickle_module=None, 
    registered_model_name=None, 
    **kwargs
    )
```

구체적인 예시는 다음과 같습니다.

```
with mlflow.start_run():

    ...

    mlflow.pytorch.log_model(model, "my_model")
```

학습에 사용된 모델 model을 my_model 이라는 이름으로 저장하게 됩니다. 구체적으로는 해당 run의 `artifacts` 디렉토리 내에 `my_model/data`경로의 디렉토리가 생성되고, 내부에 model이 저장됩니다. 또한 model과 관련하여 conda.yaml, MLmodel 파일이 생성됩니다. 정확한 디렉토리 구조는 다음과 같습니다.

```
mlruns
|- .trash
|- 0   // experiment id
    |- ...b0ffc465381dbfb77995d231d   // run id
        |- artifacts  // 모델과 관련된 내용들이 저장되는 디렉토리
            |- my_model   // 두 번째 파라미터로 전달한 model name
                |- data
                    |- model.pth   // torch model
                    |- pickle_module_info.txt
                |- conda.yaml
                |- MLmodel
        |- metrics
        |- params
        |- tags
        |- meta.yaml
    ...
```

sklearn 등 다른 패키지를 사용하여 얻은 결과는 pickle 파일(.pkl)로 저장됩니다.

#### MLmodel

log_model method를 사용하면 MLmodel이 생성됩니다. 구체적인 내용은 다음과 같습니다.

```
<MLmodel>

artifact_path: my_model
flavors:
  python_function:
    data: data
    env: conda.yaml
    loader_module: mlflow.pytorch
    pickle_module_name: mlflow.pytorch.pickle_module
    python_version: 3.6.8
  pytorch:
    model_data: data
    pytorch_version: 1.4.0
run_id: a692b874a5464d9ea0d822b42fcd****
utc_time_created: '2020-02-18 02:57:54.437482'
```

MLmodel 파일은 모델을 deploy 할 때 사용됩니다.

### 모델 deploy

생성된 모델을 deploy하기 위해서는 아래와 같은 command line을 사용합니다.

```
$ mlflow models serve -m [model dir] [options]
```

`-m` 또는 `--model-uri` 옵션으로 MLmodel 파일이 저장되어 있는 디렉토리를 설정해주면 됩니다.

이외 다음과 같은 옵션이 있습니다.

- port

    port를 설정할 수 있습니다. 기본값은 5000입니다.

    ```
    -p, --port INTEGER
    ```

- host

    host를 설정할 수 있습니다. 기본값은 127.0.0.1 이고, 외부에서 접속하도록 하기 위해서는 0.0.0.0으로 바꾸어야 합니다.

    ```
    -h, --host HOST
    ```

- no conda

    ```
    --no-conda
    ```

#### CURL을 이용한 deploy 확인

deploy 여부는 `curl`을 이용해 간단히 확인할 수 있습니다. 양식은 아래와 같습니다.

```
$ curl http://127.0.0.1:5000/invocations 
    -H 'Content-Type: application/json' 
    -d '{ "columns": ["a", "b", "c"], "data": [[1, 2, 3], [4, 5, 6]] }'
```

curl에서 `-H` 옵션은 헤더를, `-d`는 POST의 대상이 되는 데이터를 말합니다. 즉 위의 command line은 `http://127.0.0.1:5000/invocations`로 주어진 데이터를 전송하라는 의미입니다. 이렇게 데이터를 전송하였을 때 model deploy가 성공적으로 되어 있다면 적절한 값을 반환합니다.
