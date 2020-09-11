---
layout: post
title: Useful Methods for Data Science
category_num: 1
subtitle: 복사 붙여넣기
---

# Useful Methods for Data Science

- Update : 2020.08.30

## pd.drop()

```python
# drop column
df = df.drop("col1", axis=1)
df = df.drop(["col2", "col3"], axis=1)
df.drop(["col4", "col5"], axis=1, inplace=True)
```

## pd.concat()

```python
df = pd.concat([df1, df2, ...])
```

## df.reset_index()

```python
df.reset_index(drop=True, inplace=True)
```

## line plot

```python
plt.figure(figsize=(10,10))
plt.plot([1, 4, 9, 16], [1,2,3,4]) # x: [1,4,9,16] y: [1,2,3,4]
plt.title("title")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

### with legend

```python
plt.figure(figsize=(10,10))
plt.plot([1, 4, 9, 16], [1,2,3,4], label="train")
plt.plot([2,3,4,5], [1,2,3,4], label="test")
plt.title("title")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

#### beautiful legend

- legend under the plot 

```python
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),fancybox=True, shadow=False, ncol=5)
```

### with marker

- Not dot plot

```python
plt.figure(figsize=(10,10))
plt.plot([1, 4, 9, 16], [1,2,3,4], label="train", marker=".")
plt.plot([2,3,4,5], [1,2,3,4], label="test", marker=".")
plt.title("title")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

## dot plot

```python
plt.figure(figsize=(10,10))
plt.plot([1, 4, 9, 16], [1,2,3,4], ".") # x: [1,4,9,16] y: [1,2,3,4]
plt.title("title")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

### small dot plot

```python
plt.figure(figsize=(10,10))
plt.plot([1, 4, 9, 16], [1,2,3,4], ",") # x: [1,4,9,16] y: [1,2,3,4]
plt.title("title")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

## save image

```python
plt.savefig(file.png, dpi=100, bbox_inches='tight')
```

## Matplotlib Font Setting

```python
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)
```

## PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(df)

df_pca = pca.transform(df)
df1_pca = pca.transform(df1)

plt.figure(figsize=(10,10))
plt.plot(df_pca[:, 0], df_pca[:, 1], label="df")
plt.plot(df1_pca[:, 0], df1_pca[:, 1], label="df1")
plt.title("PCA")
plt.legend()
plt.show()
```
