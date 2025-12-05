### 리스트

**코드:**
```python
list_a = ["John", 33,"Toronto", True]
list_a
```

**결과:**
```
['John', 33, 'Toronto', True]
```

---

**코드:**
```python
type(list_a)
```

**결과:**
```
list
```

---

**코드:**
```python
bin_colors=['Red','Green','Blue','Yellow']
bin_colors[1]
```

**결과:**
```
'Green'
```

#### 슬라이싱

**코드:**
```python
bin_colors[0:2]
```

**결과:**
```
['Red', 'Green']
```

---

**코드:**
```python
bin_colors[1]
```

**결과:**
```
'Green'
```

---

**코드:**
```python
bin_colors[2:]
```

**결과:**
```
['Blue', 'Yellow']
```

---

**코드:**
```python
bin_colors[:2]
```

**결과:**
```
['Red', 'Green']
```

#### 음수 인덱스

**코드:**
```python
bin_colors[:-1]
```

**결과:**
```
['Red', 'Green', 'Blue']
```

---

**코드:**
```python
bin_colors[:-2]
```

**결과:**
```
['Red', 'Green']
```

---

**코드:**
```python
bin_colors[-2:-1]
```

**결과:**
```
['Blue']
```

#### 중첩

**코드:**
```python
a = [1,2,[100,200,300],6]
max(a[2])
```

**결과:**
```
300
```

---

**코드:**
```python
a[2][1]
```

**결과:**
```
200
```

#### 반복

**코드:**
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

---

**코드:**
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

### 튜플

**코드:**
```python
bin_colors=('Red','Green','Blue','Yellow')
print(f"The second element of the tuple is {bin_colors[1]}")
```

**결과:**
```
The second element of the tuple is Green
```

---

**코드:**
```python
print(f"The elements after thrid element onwards are {bin_colors[2:]}")
```

**결과:**
```
The elements after thrid element onwards are ('Blue', 'Yellow')
```

---

**코드:**
```python
nested_tuple = (1,2,(100,200,300),6)
print(f"The maximum value of the inner tuple {max(nested_tuple[2])}")
```

**결과:**
```
The maximum value of the inner tuple 300
```

### 딕셔너리

**코드:**
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

---

**코드:**
```python
bin_colors.get('approved_color')
```

**결과:**
```
'Green'
```

---

**코드:**
```python
bin_colors['approved_color']
```

**결과:**
```
'Green'
```

---

**코드:**
```python
bin_colors['approved_color']="Purple"
print(bin_colors)
```

**결과:**
```
{'manual_color': 'Yellow', 'approved_color': 'Purple', 'refused_color': 'Red'}
```

### 집합

**코드:**
```python
green = {'grass', 'leaves'}
print(green)
```

**결과:**
```
{'leaves', 'grass'}
```

---

**코드:**
```python
yellow = {'dandelions', 'fire hydrant', 'leaves'}
red = {'fire hydrant', 'blood', 'rose', 'leaves'}
print(f"The union of yellow and red sets is {yellow|red}")
```

**결과:**
```
The union of yellow and red sets is {'dandelions', 'leaves', 'rose', 'fire hydrant', 'blood'}
```

---

**코드:**
```python
print(f"The intersaction of yellow and red is {yellow&red}")
```

**결과:**
```
The intersaction of yellow and red is {'fire hydrant', 'leaves'}
```

### 데이터프레임

**코드:**
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

#### 열 선택

**코드:**
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

**코드:**
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

#### 행 선택

**코드:**
```python
df.iloc[1:3,:]
```

**결과:**
```
id	name	age	decision
1	2	Elena	23	False
2	3	Steven	40	True
```

---

**코드:**
```python
df[df.age>30]
```

**결과:**
```
id	name	age	decision
0	1	Fares	32	True
2	3	Steven	40	True
```

---

**코드:**
```python
df[(df.age<35)&(df.decision==True)]
```

**결과:**
```
id	name	age	decision
0	1	Fares	32	True
```

## 2. 스택

**코드:**
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

**코드:**
```python
stack=Stack()
stack.push('Red')
stack.push('Green')
stack.push("Blue")
stack.push("Yellow")
```

### Pop

**코드:**
```python
stack.pop()
```

**결과:**
```
'Yellow'
```

---

**코드:**
```python
stack.isEmpty()
```

**결과:**
```
False
```

---

**코드:**
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

## 큐

**코드:**
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

---

**코드:**
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

---

**코드:**
```python
print(queue.dequeue())
```

**결과:**
```
Red
```