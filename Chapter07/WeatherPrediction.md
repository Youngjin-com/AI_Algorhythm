# Chapter 7

# 로지스틱 회귀 - 날씨 예측  

## 라이브러리 임포트 및 데이터 로드

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv("https://storage.googleapis.com/neurals/data/weather.csv")
df.head()
```

**결과:**
```
Date	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	WindDir3pm	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RISK_MM	RainTomorrow
0	2007-11-01	8.0	24.3	0.0	3.4	6.3	NW	30.0	SW	NW	...	29	1019.7	1015.0	7	7	14.4	23.6	No	3.6	Yes
1	2007-11-02	14.0	26.9	3.6	4.4	9.7	ENE	39.0	E	W	...	36	1012.4	1008.4	5	3	17.5	25.7	Yes	3.6	Yes
2	2007-11-03	13.7	23.4	3.6	5.8	3.3	NW	85.0	N	NNE	...	69	1009.5	1007.2	8	7	15.4	20.2	Yes	39.8	Yes
3	2007-11-04	13.3	15.5	39.8	7.2	9.1	NW	54.0	WNW	W	...	56	1005.5	1007.0	2	7	13.5	14.1	Yes	2.8	Yes
4	2007-11-05	7.6	16.1	2.8	5.6	10.6	SSE	50.0	SSE	ESE	...	49	1018.3	1018.5	7	7	11.1	15.4	Yes	0.0	No
5 rows × 23 columns
```

---

## 기술 통계

```python
df.describe()
```

**결과:**
```
MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustSpeed	WindSpeed9am	WindSpeed3pm	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RISK_MM
count	366.000000	366.000000	366.000000	366.000000	363.000000	364.000000	359.000000	366.000000	366.000000	366.000000	366.000000	366.000000	366.000000	366.000000	366.000000	366.000000	366.000000
mean	7.265574	20.550273	1.428415	4.521858	7.909366	39.840659	9.651811	17.986339	72.035519	44.519126	1019.709016	1016.810383	3.890710	4.024590	12.358470	19.230874	1.428415
std	6.025800	6.690516	4.225800	2.669383	3.481517	13.059807	7.951929	8.856997	13.137058	16.850947	6.686212	6.469422	2.956131	2.666268	5.630832	6.640346	4.225800
min	-5.300000	7.600000	0.000000	0.200000	0.000000	13.000000	0.000000	0.000000	36.000000	13.000000	996.500000	996.800000	0.000000	0.000000	0.100000	5.100000	0.000000
25%	2.300000	15.025000	0.000000	2.200000	5.950000	31.000000	6.000000	11.000000	64.000000	32.250000	1015.350000	1012.800000	1.000000	1.000000	7.625000	14.150000	0.000000
50%	7.450000	19.650000	0.000000	4.200000	8.600000	39.000000	7.000000	17.000000	72.000000	43.000000	1020.150000	1017.400000	3.500000	4.000000	12.550000	18.550000	0.000000
75%	12.500000	25.500000	0.200000	6.400000	10.500000	46.000000	13.000000	24.000000	81.000000	55.000000	1024.475000	1021.475000	7.000000	7.000000	17.000000	24.000000	0.200000
max	20.900000	35.800000	39.800000	13.800000	13.600000	98.000000	41.000000	52.000000	99.000000	96.000000	1035.700000	1033.200000	8.000000	8.000000	24.700000	34.500000	39.800000
```

---

## 데이터 크기

```python
df.size
```

**결과:**
```
8418
```

---

## 데이터 형태

```python
df.shape
```

**결과:**
```
(366, 23)
```

---

## 컬럼 확인

```python
df.columns
```

**결과:**
```
Index(['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'],
      dtype='object')
```

---

## 풍향 9am 시각화

```python
df.WindDir9am.value_counts().plot(kind = 'barh')
plt.title("Wind Direction 9 am")
plt.show()
```

**결과:**
```
No description has been provided for this image
```

---

## 풍향 3pm 시각화

```python
df.WindDir3pm.value_counts().plot(kind="barh")
plt.title("Wind Direction 3 PM")
plt.show()
```

**결과:**
```
No description has been provided for this image
```

---

## RainToday 및 RainTomorrow 변환

```python
df['RainToday']=df['RainToday'].apply(lambda x:1 if x == "Yes" else 0)
df['RainTomorrow']=df['RainTomorrow'].apply(lambda x:1 if x == "Yes" else 0)
df.head()
```

**결과:**
```
Date	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	WindDir3pm	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RISK_MM	RainTomorrow
0	2007-11-01	8.0	24.3	0.0	3.4	6.3	NW	30.0	SW	NW	...	29	1019.7	1015.0	7	7	14.4	23.6	0	3.6	1
1	2007-11-02	14.0	26.9	3.6	4.4	9.7	ENE	39.0	E	W	...	36	1012.4	1008.4	5	3	17.5	25.7	1	3.6	1
2	2007-11-03	13.7	23.4	3.6	5.8	3.3	NW	85.0	N	NNE	...	69	1009.5	1007.2	8	7	15.4	20.2	1	39.8	1
3	2007-11-04	13.3	15.5	39.8	7.2	9.1	NW	54.0	WNW	W	...	56	1005.5	1007.0	2	7	13.5	14.1	1	2.8	1
4	2007-11-05	7.6	16.1	2.8	5.6	10.6	SSE	50.0	SSE	ESE	...	49	1018.3	1018.5	7	7	11.1	15.4	1	0.0	0
5 rows × 23 columns
```

---

## 돌풍 방향 시각화

```python
df.WindGustDir.value_counts().plot(kind = "barh",color = 'c')
plt.title("Wind Gust Direction")
plt.show()
```

**결과:**
```
No description has been provided for this image
```

---

## 레이블 인코딩 및 결측치 제거

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df=df.dropna()
df.shape
```

**결과:**
```
(328, 23)
```

---

## 범주형 변수 인코딩

```python
df.WindGustDir = le.fit_transform(df.WindGustDir)
df.WindDir3pm = le.fit_transform(df.WindDir3pm)
df.WindDir9am = le.fit_transform(df.WindDir9am)
df.columns
```

**결과:**
```
Index(['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'],
      dtype='object')
```

---

## 인코딩 후 기술 통계

```python
df.describe()
```

**결과:**
```
MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	WindDir3pm	WindSpeed9am	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RISK_MM	RainTomorrow
count	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	...	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000	328.000000
mean	7.742988	20.897561	1.440854	4.702439	8.014939	6.192073	40.396341	7.067073	7.512195	10.414634	...	44.003049	1019.350000	1016.530793	3.905488	4.000000	12.815549	19.556402	0.185976	1.422561	0.182927
std	5.945199	6.707310	4.289427	2.681183	3.506646	4.337765	13.132176	3.897197	4.560819	7.811544	...	16.605975	6.715244	6.469774	2.974957	2.652101	5.542521	6.644311	0.389681	4.234023	0.387197
min	-5.300000	7.600000	0.000000	0.200000	0.000000	0.000000	13.000000	0.000000	0.000000	2.000000	...	13.000000	996.500000	996.800000	0.000000	0.000000	0.100000	5.100000	0.000000	0.000000	0.000000
25%	2.850000	15.500000	0.000000	2.550000	6.000000	2.000000	31.000000	3.000000	4.000000	6.000000	...	32.000000	1014.800000	1012.400000	1.000000	1.000000	8.175000	14.500000	0.000000	0.000000	0.000000
50%	7.900000	20.400000	0.000000	4.400000	8.750000	6.500000	39.000000	7.500000	7.000000	7.000000	...	42.500000	1019.750000	1016.900000	4.000000	4.000000	13.500000	18.850000	0.000000	0.000000	0.000000
75%	12.800000	25.800000	0.200000	6.600000	10.700000	8.000000	46.000000	10.000000	13.000000	13.000000	...	54.000000	1024.300000	1021.125000	7.000000	7.000000	17.200000	24.225000	0.000000	0.200000	0.000000
max	20.900000	35.800000	39.800000	13.800000	13.600000	15.000000	98.000000	15.000000	15.000000	41.000000	...	93.000000	1035.700000	1033.200000	8.000000	8.000000	24.700000	34.500000	1.000000	39.800000	1.000000
8 rows × 22 columns
```

---

## 데이터 분할

```python
from sklearn.model_selection import train_test_split
x = df.drop(['Date','RainTomorrow'],axis=1)
y = df['RainTomorrow']
train_x , train_y ,test_x , test_y = train_test_split(x,y , test_size = 0.2,random_state = 2)
train_x.shape
```

**결과:**
```
(262, 21)
```

```python
train_y.shape
```

**결과:**
```
(66, 21)
```

---

## 로지스틱 회귀 모델 학습

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x , test_x)
```

**결과:**
```
/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

LogisticRegression
LogisticRegression()
```

---

## 예측 및 정확도 평가

```python
predict = model.predict(train_y)
from sklearn.metrics import accuracy_score
accuracy_score(predict , test_y)
```

**결과:**
```
0.9696969696969697
```