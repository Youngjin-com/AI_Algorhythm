# Chapter 7

# 나이브 베이즈 알고리즘 (Naive Bayes Algorithm)

## 라이브러리 임포트

```python
# Importing the libraries
import numpy as np
import sklearn,sklearn.tree
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
print(sklearn.__version__)
```

**결과:**
```
1.2.2
```

---

## 데이터셋 임포트

```python
# Importing the dataset
dataset = pd.read_csv('https://storage.googleapis.com/neurals/data/data/Social_Network_Ads.csv')
dataset = dataset.drop(columns=['User ID'])
```

---

## 데이터 확인

```python
dataset.head(5)
```

**결과:**
```
GenderAgeEstimatedSalaryPurchased0Male191900001Male352000002Female264300003Female275700004Male19760000
```

---

## 원-핫 인코딩

```python
enc = sklearn.preprocessing.OneHotEncoder()
enc.fit(dataset.iloc[:,[0]])
onehotlabels = enc.transform(dataset.iloc[:,[0]]).toarray()
genders = pd.DataFrame({'Female': onehotlabels[:, 0], 'Male': onehotlabels[:, 1]})
result = pd.concat([genders,dataset.iloc[:,1:]], axis=1, sort=False)
result.head(5)
```

**결과:**
```
FemaleMaleAgeEstimatedSalaryPurchased00.01.01919000010.01.03520000021.00.02643000031.00.02757000040.01.019760000
```

---

## 특징과 레이블 분리

```python
y=result['Purchased']
X=result.drop(columns=['Purchased'])
```

---

## 훈련 세트와 테스트 세트 분할

```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

---

## 특징 스케일링

```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

## 나이브 베이즈 분류기 학습

```python
# Fitting Decision Tree Classification to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

**결과:**
```
GaussianNB

GaussianNB()
```

---

## 테스트 세트 예측

```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
cm
```

**결과:**
```
array([[66,  2],
       [ 6, 26]])
```

---

## 성능 평가

```python
accuracy= metrics.accuracy_score(y_test,y_pred)
recall = metrics.recall_score(y_test,y_pred)
precision = metrics.precision_score(y_test,y_pred)
print(accuracy,recall,precision)
```

**결과:**
```
0.92 0.8125 0.9285714285714286
```