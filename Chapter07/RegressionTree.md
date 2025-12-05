# Chapter 7

# 의사결정 트리 회귀 (Decision Tree Regression)

## 라이브러리 임포트

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

---

## 데이터셋 임포트

```python
# Importing the dataset
dataset = pd.read_csv('https://storage.googleapis.com/neurals/data/data/auto.csv')
dataset.head(5)
```

**결과:**
```
NAMECYLINDERSDISPLACEMENTHORSEPOWERWEIGHTACCELERATIONMPG0chevrolet chevelle malibu8307.0130350412.018.01buick skylark 3208350.0165369311.515.02plymouth satellite8318.0150343611.018.03amc rebel sst8304.0150343312.016.04ford torino8302.0140344910.517.0
```

---

## 데이터 전처리

```python
dataset=dataset.drop(columns=['NAME'])
dataset= dataset.apply(pd.to_numeric, errors='coerce')
dataset.fillna(0, inplace=True)
dataset.head(5)
```

**결과:**
```
CYLINDERSDISPLACEMENTHORSEPOWERWEIGHTACCELERATIONMPG08307.0130.0350412.018.018350.0165.0369311.515.028318.0150.0343611.018.038304.0150.0343312.016.048302.0140.0344910.517.0
```

---

## 특징과 레이블 분리 및 데이터 분할

```python
y=dataset['MPG']
X=dataset.drop(columns=['MPG'])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

---

## 의사결정 트리 회귀 모델 학습

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X_train, y_train)
```

**결과:**
```
DecisionTreeRegressor

DecisionTreeRegressor(max_depth=3)
```

---

## 테스트 세트 예측

```python
# Predicting the Test set results
y_pred = regressor.predict(X_test)
```

---

## 성능 평가 (RMSE)

```python
from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(y_test, y_pred))
```

**결과:**
```
4.464255966462035
```