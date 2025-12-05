# Chapter 7

# 그래디언트 부스팅 (Gradient Boosting)

## 라이브러리 임포트

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
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
dataset.head(5)
dataset= dataset.apply(pd.to_numeric, errors='coerce')
dataset.fillna(0, inplace=True)
dataset
```

**결과:**
```
CYLINDERSDISPLACEMENTHORSEPOWERWEIGHTACCELERATIONMPG08307.0130.0350412.018.018350.0165.0369311.515.028318.0150.0343611.018.038304.0150.0343312.016.048302.0140.0344910.517.0.....................3934140.086.0279015.627.0394497.052.0213024.644.03954135.084.0229511.632.03964120.079.0262518.628.03974119.082.0272019.431.0
398 rows × 6 columns
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

## 그래디언트 부스팅 회귀 모델 학습

```python
from sklearn import ensemble

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'squared_error'}
regressor = ensemble.GradientBoostingRegressor(**params)

regressor.fit(X_train, y_train)
```

**결과:**
```
GradientBoostingRegressor

GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=500)
```

---

## 테스트 세트 예측

```python
# Predicting the Test set result
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
4.039759805419003
```