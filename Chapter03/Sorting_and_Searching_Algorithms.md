# Chapter 3

# 정렬 및 검색 알고리즘 (Sorting and Searching Algorithms)

## Python에서의 교환 함수 (Swap Function in Python)

```python
var_1 = 1
var_2 = 2
var_1,var_2 = var_2,var_1
print(var_1,var_2)
```

**결과:**
```
2 1
```

---

## 정렬 알고리즘 (Sorting Algorithms)

### 버블 정렬의 패스 1 (Pass 1 of Bubble Sort)

```python
list = [25,21,22,24,23,27,26]

last_element_index = len(list) - 1
print(0, list)
for idx in range(last_element_index):
    if list[idx] > list[idx + 1]:
        list[idx], list[idx + 1] = list[idx + 1], list[idx]
    print(idx + 1, list)
```

**결과:**
```
0 [25, 21, 22, 24, 23, 27, 26]
1 [21, 25, 22, 24, 23, 27, 26]
2 [21, 22, 25, 24, 23, 27, 26]
3 [21, 22, 24, 25, 23, 27, 26]
4 [21, 22, 24, 23, 25, 27, 26]
5 [21, 22, 24, 23, 25, 27, 26]
6 [21, 22, 24, 23, 25, 26, 27]
```

```python
list
```

**결과:**
```
[21, 22, 24, 23, 25, 26, 27]
```

---

### 버블 정렬 알고리즘 (Bubble Sort Algorithm)

```python
def bubble_sort(list):
# Excahnge the elements to arrange in order
    last_element_index = len(list)-1
    for pass_no in range(last_element_index,0,-1):
        for idx in range(pass_no):
            if list[idx]>list[idx+1]:
                list[idx],list[idx+1]=list[idx+1],list[idx]
    return list

list = [25,21,22,24,23,27,26]
bubble_sort(list)
```

**결과:**
```
[21, 22, 23, 24, 25, 26, 27]
```

---

### 버블 정렬 최적화 (Optimizating bubble sort)

```python
def optimized_bubble_sort(list):
    last_element_index = len(list)-1
    for pass_no in range(last_element_index, 0, -1):
        swapped = False
        for idx in range(pass_no):
            if list[idx] > list[idx+1]:
                list[idx], list[idx+1] = list[idx+1], list[idx]
                swapped = True
        if not swapped:
            break
    return list

list = [25,21,22,24,23,27,26]
optimized_bubble_sort(list)
```

**결과:**
```
[21, 22, 23, 24, 25, 26, 27]
```

---

### 삽입 정렬 (Insertion Sort)

```python
def insertion_sort(elements):
    for i in range(1, len(elements)):
        j = i - 1
        next_element = elements[i]

        # Iterate backward through the sorted portion,
        # looking for the appropriate position for 'next_element'
        while j >= 0 and elements[j] > next_element:
            elements[j + 1] = elements[j]
            j -= 1

        elements[j + 1] = next_element
    return elements

insertion_sort(list)
```

**결과:**
```
[21, 22, 23, 24, 25, 26, 27]
```

---

### 병합 정렬 (Merge Sort)

```python
def merge_sort(elements):
    # Base condition to break the recursion
    if len(elements) <= 1:
        return elements

    mid = len(elements) // 2  # Split the list in half
    left = elements[:mid]
    right = elements[mid:]

    merge_sort(left)   # Sort the left half
    merge_sort(right)  # Sort the right half

    a, b, c = 0, 0, 0
    # Merge the two halves
    while a < len(left) and b < len(right):
        if left[a] < right[b]:
            elements[c] = left[a]
            a += 1
        else:
            elements[c] = right[b]
            b += 1
        c += 1

    # If there are remaining elements in the left half
    while a < len(left):
        elements[c] = left[a]
        a += 1
        c += 1
    # If there are remaining elements in the right half
    while b < len(right):
        elements[c] = right[b]
        b += 1
        c += 1
    return elements

list = [21, 22, 23, 24, 25, 26, 27]
merge_sort(list)
```

**결과:**
```
[21, 22, 23, 24, 25, 26, 27]
```

---

### 셸 정렬 (Shell Sort)

```python
def shell_sort(elements):
    distance = len(elements) // 2
    while distance > 0:
        for i in range(distance, len(elements)):
            temp = elements[i]
            j = i
# Sort the sub list for this distance
            while j >= distance and elements[j - distance] > temp:
                list[j] = elements[j - distance]
                j = j-distance
            list[j] = temp
# Reduce the distance for the next element
        distance = distance//2
    return elements

list = [21, 22, 23, 24, 25, 26, 27]
shell_sort(list)
```

**결과:**
```
[21, 22, 23, 24, 25, 26, 27]
```

---

### 선택 정렬 (Selection Sort)

```python
def selection_sort(list):
    for fill_slot in range(len(list) - 1, 0, -1):
        max_index = 0
        for location in range(1, fill_slot + 1):
            if list[location] > list[max_index]:
                max_index = location
        list[fill_slot],list[max_index] = list[max_index],list[fill_slot]
    return list

list = [21, 22, 23, 24, 25, 26, 27]
selection_sort(list)
```

**결과:**
```
[21, 22, 23, 24, 25, 26, 27]
```

---

## 검색 알고리즘 (Searching Algorithms)

### 선형 검색 (Linear Search)

```python
def linear_search(list, item):
    index = 0
    found = False

# Match the value with each data element
    while index < len(list) and found is False:
        if list[index] == item:
            found = True
        else:
            index = index + 1
    return found

list = [12, 33, 11, 99, 22, 55, 90]
print(linear_search(list, 12))
print(linear_search(list, 91))
```

**결과:**
```
True
False
```

---

### 이진 검색 (Binary Search)

```python
def binary_search(elements, item):
    first = 0
    last = len(elements) - 1

    while first <= last:
        midpoint = (first + last) // 2
        if elements[midpoint] == item:
            return True
        else:
            if item < elements[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
    return False

list = [12, 33, 11, 99, 22, 55, 90]
sorted_list = bubble_sort(list)
print(binary_search(list, 12))
print(binary_search(list, 91))
```

**결과:**
```
True
False
```

---

### 보간 검색 (Intpolation Search)

```python
def int_polsearch(list,x ):
    idx0 = 0
    idxn = (len(list) - 1)
    while idx0 <= idxn and x >= list[idx0] and x <= list[idxn]:

# Find the mid point
        mid = idx0 +int(((float(idxn - idx0)/( list[idxn] - list[idx0])) * ( x - list[idx0])))

# Compare the value at mid point with search value
        if list[mid] == x:
            return True
        if list[mid] < x:
            idx0 = mid + 1
    return False

list = [12, 33, 11, 99, 22, 55, 90]
sorted_list = bubble_sort(list)
print(int_polsearch(list, 12))
print(int_polsearch(list,91))
```

**결과:**
```
True
False
```
