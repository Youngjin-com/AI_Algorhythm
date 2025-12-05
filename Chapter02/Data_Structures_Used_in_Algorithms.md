# CHAPTER 2

# 알고리즘에 사용되는 데이터 구조 (Data Structures Used in Algorithms)

## 1 Python에서 데이터 구조 탐색하기 (Exploring data structures in Python)

### 1.1 리스트 (Lists)

```python
list_a = ["John", 33,"Toronto", True]
list_a
```

**결과:**
```
['John', 33, 'Toronto', True]
```

```python
type(list_a)
```

**결과:**
```
list
```

```python
bin_colors=['Red','Green','Blue','Yellow']
bin_colors[1]
```

**결과:**
```
'Green'
```

---

### 1.1.2 슬라이싱 (Slicing)

```python
bin_colors[0:2]
```

**결과:**
```
['Red', 'Green']
```

```python
bin_colors[1]
```

**결과:**
```
'Green'
```

```python
bin_colors[2:]
```

**결과:**
```
['Blue', 'Yellow']
```

```python
bin_colors[:2]
```

**결과:**
```
['Red', 'Green']
```

---

### 1.1.3 음수 인덱스 (Negative Indices)

```python
bin_colors[:-1]
```

**결과:**
```
['Red', 'Green', 'Blue']
```

```python
bin_colors[:-2]
```

**결과:**
```
['Red', 'Green']
```

```python
bin_colors[-2:-1]
```

**결과:**
```
['Blue']
```

---

### 1.1.4 중첩 (Nesting)

```python
a = [1,2,[100,200,300],6]
max(a[2])
```

**결과:**
```
300
```

```python
a[2][1]
```

**결과:**
```
200
```

---

### 1.1.5 반복 (Iteration)

```python
for aColor in bin_colors:
    print(aColor+ " Square")
```

**결과:**
```
Red Square
Green Square
Blue Square
Yellow Square
```

```python
numbers = [1,2,3]
letters = ['a','b','c']
combined = zip (numbers,letters)
combined_list = list(combined)
combined_list
```

**결과:**
```
[(1, 'a'), (2, 'b'), (3, 'c')]
```

---

## 튜플 (Tuples)

```python
bin_colors=('Red','Green','Blue','Yellow')
print(f"The second element of the tuple is {bin_colors[1]}")
```

**결과:**
```
The second element of the tuple is Green
```

```python
print(f"The elements after thrid element onwards are {bin_colors[2:]}")
```

**결과:**
```
The elements after thrid element onwards are ('Blue', 'Yellow')
```

```python
nested_tuple = (1,2,(100,200,300),6)
print(f"The maximum value of the inner tuple {max(nested_tuple[2])}")
```

**결과:**
```
The maximum value of the inner tuple 300
```

---

## 1.2 딕셔너리 (Dictionary)

```python
bin_colors ={
  "manual_color": "Yellow",
  "approved_color": "Green",
  "refused_color": "Red"
}
print(bin_colors)
```

**결과:**
```
{'manual_color': 'Yellow', 'approved_color': 'Green', 'refused_color': 'Red'}
```

```python
bin_colors.get('approved_color')
```

**결과:**
```
'Green'
```

```python
bin_colors['approved_color']
```

**결과:**
```
'Green'
```

```python
bin_colors['approved_color']="Purple"
print(bin_colors)
```

**결과:**
```
{'manual_color': 'Yellow', 'approved_color': 'Purple', 'refused_color': 'Red'}
```

---

## 1.3 집합 (Set)

```python
green = {'grass', 'leaves'}
print(green)
```

**결과:**
```
{'leaves', 'grass'}
```

```python
yellow = {'dandelions', 'fire hydrant', 'leaves'}
red = {'fire hydrant', 'blood', 'rose', 'leaves'}
print(f"The union of yellow and red sets is {yellow|red}")
```

**결과:**
```
The union of yellow and red sets is {'dandelions', 'leaves', 'rose', 'fire hydrant', 'blood'}
```

```python
print(f"The intersaction of yellow and red is {yellow&red}")
```

**결과:**
```
The intersaction of yellow and red is {'fire hydrant', 'leaves'}
```

---

## 1.4 데이터프레임 (DataFrame)

```python
import pandas as pd
df = pd.DataFrame([
    ['1', 'Fares', 32, True],
    ['2', 'Elena', 23, False],
    ['3', 'Steven', 40, True]])
df.columns = ['id', 'name', 'age', 'decision']
print(df)
```

**결과:**
```
  id    name  age  decision
0  1   Fares   32      True
1  2   Elena   23     False
2  3  Steven   40      True
```

---

### 1.4.1 열 선택 (Column Selection)

```python
df[['name','age']]
```

**결과:**
```
name	age
0	Fares	32
1	Elena	23
2	Steven	40
```

---

## 값 (Values)

**결과:**
```
No description has been provided for this image
```

---

## 분포 (Distributions)

**결과:**
```
No description has been provided for this image
```

---

## 범주형 분포 (Categorical distributions)

**결과:**
```
No description has been provided for this image
```

---

## 패싯 분포 (Faceted distributions)

**결과:**
```
No description has been provided for this image
```

---

## 위치별 열 선택

```python
df.iloc[:,3]
```

**결과:**
```
0     True
1    False
2     True
Name: decision, dtype: bool
```

---

### 1.4.2 행 선택 (Row Selection)

```python
df.iloc[1:3,:]
```

**결과:**
```
id	name	age	decision
1	2	Elena	23	False
2	3	Steven	40	True
```

```python
df[df.age>30]
```

**결과:**
```
id	name	age	decision
0	1	Fares	32	True
2	3	Steven	40	True
```

```python
df[(df.age<35)&(df.decision==True)]
```

**결과:**
```
id	name	age	decision
0	1	Fares	32	True
```

---

## 2 스택 (Stack)

```python
class Stack:
     def __init__(self):
         self.items = []
     def isEmpty(self):
         return self.items == []
     def push(self, item):
         self.items.append(item)
     def pop(self):
         return self.items.pop()
     def peek(self):
         return self.items[len(self.items)-1]
     def size(self):
         return len(self.items)
```

---

## 스택에 요소 추가 (Populate the stack)

```python
stack=Stack()
stack.push('Red')
stack.push('Green')
stack.push("Blue")
stack.push("Yellow")
```

---

## Pop

```python
stack.pop()
```

**결과:**
```
'Yellow'
```

```python
stack.isEmpty()
```

**결과:**
```
False
```

```python
colors = ['Red']
colors.append('Green')
colors.append('Yellow')
colors.append('Blue')
colors
```

**결과:**
```
['Red', 'Green', 'Yellow', 'Blue']
```

---

## 큐 (Queue)

```python
class Queue(object):
   def __init__(self):
      self.items = []
   def isEmpty(self):
      return self.items == []
   def enqueue(self, item):
       self.items.insert(0,item)
   def dequeue(self):
      return self.items.pop()
   def size(self):
      return len(self.items)
```

```python
queue = Queue()
queue.enqueue("Red")
queue.enqueue('Green')
queue.enqueue('Blue')
queue.enqueue('Yellow')
print(f"Size of queue is {queue.size()}")
```

**결과:**
```
Size of queue is 4
```

```python
print(queue.dequeue())
```

**결과:**
```
Red
```