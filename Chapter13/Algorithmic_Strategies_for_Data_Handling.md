# Chapter 13

# 데이터 처리를 위한 알고리즘 전략  

## Python에서 허프만 코딩 구현하기  

### Node 클래스 정의

```python
import heapq
import functools

@functools.total_ordering
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        return self.freq == other.freq
```

---

## 허프만 트리 구축 함수  

```python
def build_tree(frequencies):
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]  # the root node
```

---

## 예제 사용  

```python
frequencies = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
root = build_tree(frequencies)
print(root.freq)
```

**결과:**
```
100
```

---

## 트리 순회를 통한 허프만 코드 생성  

```python
import heapq
import functools

@functools.total_ordering
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        return self.freq == other.freq

def build_tree(frequencies):
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]  # the root node

def generate_codes(node, code='', codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return {}
    if node.char is not None:
        codes[node.char] = code
        return codes
    generate_codes(node.left, code + '0', codes)
    generate_codes(node.right, code + '1', codes)
    return codes
```

---

## 허프만 인코딩을 위한 샘플 데이터 

```python
data = {
    'L': 0.45,
    'M': 0.13,
    'N': 0.12,
    'X': 0.16,
    'Y': 0.09,
    'Z': 0.05
}
```

---

## 허프만 트리 구축 및 코드 생성  

```python
root = build_tree(data)
codes = generate_codes(root)
```

---

## 허프만 코드 출력  

```python
# Print the root of the Huffman tree
print(f'Root of the Huffman tree: {root}')
# Print out the Huffman codes
for char, code in codes.items():
    print(f'{char}: {code}')
```

**결과:**
```
Root of the Huffman tree: <__main__.Node object at 0x7b28f2a01570>
L: 0
N: 100
M: 101
Z: 1100
Y: 1101
X: 111
```

---

## 허프만 코드 재출력

```python
# Print the root of the Huffman tree
print(f'Root of the Huffman tree: {root}')
# Print out the Huffman codes
for char, code in codes.items():
    print(f'{char}: {code}')
```

**결과:**
```
Root of the Huffman tree: <__main__.Node object at 0x7b28f2a01570>
L: 0
N: 100
M: 101
Z: 1100
Y: 1101
X: 111
```