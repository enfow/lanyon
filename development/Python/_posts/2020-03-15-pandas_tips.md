---
layout: post
title: Pandas Tips
---

# Pandas Tips

## 1. Column의 모든 row 값 변경하기

- [참고 링크](<https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6>)
- 한 줄 요약 : pandas에서 각 row를 반복문으로 하나씩 탐색하는 것은 매우 비효율적인 만큼 `apply()` 또는 `vectorization`을 적극 활용하자.

pandas를 사용하면서 가장 쉽게 하는 실수 중 하나는 모든 row를 하나씩 탐색하도록 하는 것이다. 가장 직관적이고, list와 같은 파이썬의 자료형을 다루는 것과 유사하기 때문에 가장 쉽게 떠오르는 방법이지만, 이 경우 속도가 매우 느리다고 한다. 참고한 블로그에서는 이에 대한 대안으로 `iterrows()`, `apply()`와 같은 pandas 내장 함수를 사용하거나 vectorization 방법을 사용하는 것을 권장하고 있다.

이를 확인하기 위해 Kaggle의 [Credit Card Fraud Detection](<https://www.kaggle.com/mlg-ulb/creditcardfraud/data#>)데이터를 사용하여 간단한 실험을 진행했다.

### 1) 모든 row를 반복문으로 탐색하기

`파이썬 코드`

```python
df["V1_is_positive"] = False
start = time.time()
for idx in range(len(df["V1_is_positive"])):
    if df["V1"][idx] > 0:
        df["V1_is_positive"][idx] = True
end = time.time()
print("num of positive num : {}".format(df["V1_is_positive"].sum()))
print("operation time : {}".format(end-start))
```

`에러 코드`

```
.pyenv/versions/3.7.4/lib/python3.7/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  """

```

`출력값`

```
num of positive num : 143351
operation time : 14.673181056976318
```

우선 이렇게 하니 위와 같은 에러코드가 출력되었고, 시간은 14초 정도 걸린 것을 확인할 수 있다.

### 2) iterrows()

`파이썬 코드`

```python
v1_positive_list = []
start = time.time()
for idx, row in df.iterrows():
    v1_positive_list.append(row["V1"] > 0)
df["V1_is_positive"] = v1_positive_list
end = time.time()
print("num of positive num : {}".format(df["V1_is_positive"].sum()))
print("operation time : {}".format(end-start))
```

`출력값`

```
num of positive num : 143351
operation time : 17.821782112121582
```

iterrows()를 사용한 결과 약 17초로 1)의 방법보다 더 오래 걸렸다.

### 3) apply()

`파이썬 코드`

```python
start = time.time()
df["V1_is_positive"] = df["V1"].apply(lambda x: True
                                      if x > 0
                                      else False)
end = time.time()
print("num of positive num : {}".format(df["V1_is_positive"].sum()))
print("operation time : {}".format(end-start))
```

`출력값`

```
num of positive num : 143351
operation time : 0.07773280143737793
```

iterrows()와 비교해 볼 때 결과는 다소 충격적으로, 동일한 연산을 apply() 메소드를 사용하니 0.1초도 걸리지 않았다.

### 4) python for-in loop

`파이썬 코드`

```python
start = time.time()

def is_positive(value):
    if value > 0:
        return 1
    else:
        return 0

df["V1_is_positive"] = [is_positive(value) for value in df["V1"]]

end = time.time()
print("num of positive num : {}".format(df["V1_is_positive"].sum()))
print("operation time : {}".format(end-start))
```

`출력값`

```
num of positive num : 143351
operation time : 0.12662506103515625
```

파이썬 list 내부에서 반복문을 사용하여 출력하는 방법은 apply() 만큼은 아니지만 좋은 성능을 보여주었다.

### 5) Vectorization - np.vectorize()

`파이썬 코드`

```python
start = time.time()

def is_positive(value):
    if value > 0:
        return 1
    else:
        return 0
    
vectorized_is_positive = np.vectorize(is_positive, otypes=[np.float])

df["V1_is_positive"] = vectorized_is_positive(df["V1"])

end = time.time()
print("num of positive num : {}".format(df["V1_is_positive"].sum()))
print("operation time : {}".format(end-start))
```

`출력값`

```
num of positive num : 143351.0
operation time : 0.0626070499420166
```

if-else statement를 포함하는 함수를 vectorize하기 위해 np.vectorize() 함수를 사용했다. 속도는 apoly()와 크게 차이가 없어 보이나, [stack overflow](<https://stackoverflow.com/questions/24870953/does-pandas-iterrows-have-performance-issues>) 등에서는 항상 vectorization하는 것이 좋다고 한다.

참고로 위의 stack overflow 링크에서는 iterrows()와 동일한 결과를 내는 방법들 간의 성능을 다음과 같이 줄세우고 있다.

```
1) vectorization
2) using a custom cython routine
3) apply
    a) reductions that can be performed in cython
    b) iteration in python space
4) itertuples
5) iterrows
6) updating an empty frame (e.g. using loc one-row-at-a-time)
```
