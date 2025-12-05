# CHAPTER 6

# 비지도 머신러닝 알고리즘 

## 1 - K-means 알고리즘  

### STEP 1- 패키지 임포트  

```python
from sklearn import cluster
import pandas as pd
import numpy as np
```

---

### STEP 2- 데이터 

```python
dataset = pd.DataFrame({
    'x': [11, 11, 20, 12, 16, 33, 24, 14, 45, 52, 51, 52, 55, 53, 55, 61, 62, 70, 72, 10],
    'y': [39, 36, 30, 52, 53, 46, 55, 59, 12, 15, 16, 18, 11, 23, 14, 8, 18, 7, 24, 70]
})
```

---

### STEP 3- 모델 학습  

```python
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(dataset)
```

**결과:**
```
/usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(

KMeans
KMeans(n_clusters=2)
```

---

### STEP 4- 레이블 및 클러스터 중심 출력  

```python
labels = labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(labels)
```

**결과:**
```
[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0]
```

```python
print(centers)
```

**결과:**
```
[[16.77777778 48.88888889]
 [57.09090909 15.09090909]]
```

---

### STEP 5- 플롯 

```python
import matplotlib.pyplot as plt
plt.scatter(dataset['x'], dataset['y'], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()
```

**결과:**
```
No description has been provided for this image
```

---

## 2 계층적 알고리즘  

### STEP 1: 패키지 임포트  

```python
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
```

---

### STEP 2: 데이터프레임 생성 

```python
dataset = pd.DataFrame({
    'x': [11, 11, 20, 12, 16, 33, 24, 14, 45, 52, 51, 52, 55, 53, 55, 61, 62, 70, 72, 10],
    'y': [39, 36, 30, 52, 53, 46, 55, 59, 12, 15, 16, 18, 11, 23, 14, 8, 18, 7, 24, 70]
})
```

---

### STEP 3: 클러스터 생성  

```python
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(dataset)
```

**결과:**
```
/usr/local/lib/python3.10/dist-packages/sklearn/cluster/_agglomerative.py:983: FutureWarning: Attribute `affinity` was deprecated in version 1.2 and will be removed in 1.4. Use `metric` instead
  warnings.warn(
array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
```

---

### STEP 4: 클러스터 출력  

```python
print(cluster.labels_)
```

**결과:**
```
[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0]
```

---

## 3 DBSCAN

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
dbscan = DBSCAN(eps = 0.05, min_samples=5)
dbscan.fit(dataset)
```

**결과:**
```
DBSCAN
DBSCAN(eps=0.05)
```

```python
X, y = make_moons(n_samples=1000, noise=0.05)
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
X, y = make_moons(n_samples=1000, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
```

**결과:**
```
No description has been provided for this image
```

---

## 4 PCA

```python
from sklearn.decomposition import PCA
import pandas as pd
url = "https://storage.googleapis.com/neurals/data/iris.csv"
iris = pd.read_csv(url)

iris
```

**결과:**
```
Sepal.Length	Sepal.Width	Petal.Length	Petal.Width	Species
0	5.1	3.5	1.4	0.2	setosa
1	4.9	3.0	1.4	0.2	setosa
2	4.7	3.2	1.3	0.2	setosa
3	4.6	3.1	1.5	0.2	setosa
4	5.0	3.6	1.4	0.2	setosa
...	...	...	...	...	...
145	6.7	3.0	5.2	2.3	virginica
146	6.3	2.5	5.0	1.9	virginica
147	6.5	3.0	5.2	2.0	virginica
148	6.2	3.4	5.4	2.3	virginica
149	5.9	3.0	5.1	1.8	virginica
150 rows × 5 columns
```

```python
X = iris.drop('Species', axis=1)
pca = PCA(n_components=4)
pca.fit(X)
```

**결과:**
```
PCA
PCA(n_components=4)
```

```python
pca_df=(pd.DataFrame(pca.components_,columns=X.columns))
pca_df
```

**결과:**
```
Sepal.Length	Sepal.Width	Petal.Length	Petal.Width
0	0.361387	-0.084523	0.856671	0.358289
1	0.656589	0.730161	-0.173373	-0.075481
2	-0.582030	0.597911	0.076236	0.545831
3	-0.315487	0.319723	0.479839	-0.753657
```

```python
print(pca.explained_variance_ratio_)
```

**결과:**
```
[0.92461872 0.05306648 0.01710261 0.00521218]
```

```python
X['PC1'] = X['Sepal.Length']* pca_df['Sepal.Length'][0] + X['Sepal.Width']* pca_df['Sepal.Width'][0]+ X['Petal.Length']* pca_df['Petal.Length'][0]+X['Petal.Width']* pca_df['Petal.Width'][0]
X['PC2'] = X['Sepal.Length']* pca_df['Sepal.Length'][1] + X['Sepal.Width']* pca_df['Sepal.Width'][1]+ X['Petal.Length']* pca_df['Petal.Length'][1]+X['Petal.Width']* pca_df['Petal.Width'][1]
X['PC3'] = X['Sepal.Length']* pca_df['Sepal.Length'][2] + X['Sepal.Width']* pca_df['Sepal.Width'][2]+ X['Petal.Length']* pca_df['Petal.Length'][2]+X['Petal.Width']* pca_df['Petal.Width'][2]
X['PC4'] = X['Sepal.Length']* pca_df['Sepal.Length'][3] + X['Sepal.Width']* pca_df['Sepal.Width'][3]+ X['Petal.Length']* pca_df['Petal.Length'][3]+X['Petal.Width']* pca_df['Petal.Width'][3]
X
```

**결과:**
```
Sepal.Length	Sepal.Width	Petal.Length	Petal.Width	PC1	PC2	PC3	PC4
0	5.1	3.5	1.4	0.2	2.818240	5.646350	-0.659768	0.031089
1	4.9	3.0	1.4	0.2	2.788223	5.149951	-0.842317	-0.065675
2	4.7	3.2	1.3	0.2	2.613375	5.182003	-0.613952	0.013383
3	4.6	3.1	1.5	0.2	2.757022	5.008654	-0.600293	0.108928
4	5.0	3.6	1.4	0.2	2.773649	5.653707	-0.541773	0.094610
...	...	...	...	...	...	...	...	...
145	6.7	3.0	5.2	2.3	7.446475	5.514485	-0.454028	-0.392844
146	6.3	2.5	5.0	1.9	7.029532	4.951636	-0.753751	-0.221016
147	6.5	3.0	5.2	2.0	7.266711	5.405811	-0.501371	-0.103650
148	6.2	3.4	5.4	2.3	7.403307	5.443581	0.091399	-0.011244
149	5.9	3.0	5.1	1.8	6.892554	5.044292	-0.268943	0.188390
150 rows × 8 columns
```

---

## 5 FPGrowth

```python
!pip install pyfpgrowth
```

**결과:**
```
Requirement already satisfied: pyfpgrowth in /usr/local/lib/python3.10/dist-packages (1.0)
```

```python
import pandas as pd
import numpy as np
import pyfpgrowth as fp
```

---

### STEP 2: 트랜잭션 데이터셋 생성  

```python
dict1 = {
    'id':[0,1,2,3],
    'items':[["wickets","pads"],
    ["bat","wickets","pads","helmet"],
    ["helmet","pad"],
    ["bat","pads","helmet"]]

}
transactionSet = pd.DataFrame(dict1)
transactionSet
```

**결과:**
```
id	items
0	0	[wickets, pads]
1	1	[bat, wickets, pads, helmet]
2	2	[helmet, pad]
3	3	[bat, pads, helmet]
```

---

### STEP 4: 빈발 패턴 및 규칙 생성  

```python
patterns = fp.find_frequent_patterns(transactionSet['items'],1)
patterns
```

**결과:**
```
{('pad',): 1,
 ('helmet', 'pad'): 1,
 ('wickets',): 2,
 ('pads', 'wickets'): 2,
 ('bat', 'wickets'): 1,
 ('helmet', 'wickets'): 1,
 ('bat', 'pads', 'wickets'): 1,
 ('helmet', 'pads', 'wickets'): 1,
 ('bat', 'helmet', 'wickets'): 1,
 ('bat', 'helmet', 'pads', 'wickets'): 1,
 ('bat',): 2,
 ('bat', 'helmet'): 2,
 ('bat', 'pads'): 2,
 ('bat', 'helmet', 'pads'): 2,
 ('pads',): 3,
 ('helmet',): 3,
 ('helmet', 'pads'): 2}
```

```python
rules = fp.generate_association_rules(patterns,0.8)

rules = fp.generate_association_rules(patterns,0.3) rules

rules = fp.generate_association_rules(patterns,0.3)
rules
```

**결과:**
```
{('helmet',): (('pads',), 0.6666666666666666),
 ('pad',): (('helmet',), 1.0),
 ('pads',): (('helmet',), 0.6666666666666666),
 ('wickets',): (('bat', 'helmet', 'pads'), 0.5),
 ('bat',): (('helmet', 'pads'), 1.0),
 ('bat', 'pads'): (('helmet',), 1.0),
 ('bat', 'wickets'): (('helmet', 'pads'), 1.0),
 ('pads', 'wickets'): (('bat', 'helmet'), 0.5),
 ('helmet', 'pads'): (('bat',), 1.0),
 ('helmet', 'wickets'): (('bat', 'pads'), 1.0),
 ('bat', 'helmet'): (('pads',), 1.0),
 ('bat', 'helmet', 'pads'): (('wickets',), 0.5),s
 ('bat', 'helmet', 'wickets'): (('pads',), 1.0),
 ('bat', 'pads', 'wickets'): (('helmet',), 1.0),
 ('helmet', 'pads', 'wickets'): (('bat',), 1.0)}
```