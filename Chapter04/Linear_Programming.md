# CHAPTER 4: 알고리즘 설계 (Designing Algorithms)

## 선형 프로그래밍을 이용한 용량 계획 (Capacity Planning with Linear Programming)


## PuLP 설치

```python
#If pulp is not install then please uncomment the following line of code and run it once
!pip install pulp
```

**결과:**
```
Collecting pulp
  Downloading PuLP-2.7.0-py3-none-any.whl (14.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.3/14.3 MB 52.5 MB/s eta 0:00:00
Installing collected packages: pulp
Successfully installed pulp-2.7.0
```

---

## 라이브러리 임포트

```python
import pulp
```

---

## 문제 정의

```python
model = pulp.LpProblem("Profit_maximising_problem", pulp.LpMaximize)
```

---

## 변수 정의

```python
A = pulp.LpVariable('A', lowBound=0,  cat='Integer')
B = pulp.LpVariable('B', lowBound=0, cat='Integer')
```

---

## 목적 함수 및 제약 조건

```python
# Objective function
model += 5000 * A + 2500 * B, "Profit"

# Constraints
model += 3 * A + 2 * B <= 20
model += 4 * A + 3 * B <= 30
model += 4 * A + 3 * B <= 44
```

---

## 문제 해결

```python
# Solve our problem
model.solve()
pulp.LpStatus[model.status]
```

**결과:**
```
'Optimal'
```

---

## 결정 변수 값 출력

```python
# Print our decision variable values
print (A.varValue)
print (B.varValue)
```

**결과:**
```
6.0
1.0
```

---

## 목적 함수 값 출력

```python
# Print our objective function value
print (pulp.value(model.objective))
```

**결과:**
```
32500.0
```