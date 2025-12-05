# PageRank 알고리즘  

## 필요한 라이브러리 임포트  

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
```

---

## 방향 그래프 생성 및 노드와 엣지 추가  

```python
my_web = nx.DiGraph()
my_pages = range(1,6)  # Changed to 6 since you had 5 pages from 1 to 5
connections = [(1,3),(2,1),(2,3),(3,1),(3,2),(3,4),(4,5),(5,1),(5,4)]
my_web.add_nodes_from(my_pages)
my_web.add_edges_from(connections)
```

---

## 그래프 시각화  

```python
pos = nx.shell_layout(my_web)
nx.draw(my_web, pos, arrows=True, with_labels=True)
plt.show()
```

---

## 전이 행렬 생성 함수  

```python
def create_page_rank(a_graph):
    nodes_set = len(a_graph)
    M = nx.to_numpy_array(a_graph)

    outwards = np.squeeze(np.asarray(np.sum(M, axis=1)))
    prob_outwards = np.array([1.0 / count if count > 0 else 0.0 for count in outwards])

    G = np.asarray(np.multiply(M.T, prob_outwards))
    p = np.ones(nodes_set) / float(nodes_set)
    return G, p
```

---

## 그래프의 전이 행렬 생성  

```python
G, p = create_page_rank(my_web)
print(G)
```

**결과:**
```
[[0.         0.5        0.33333333 0.         0.5       ]
 [0.         0.         0.33333333 0.         0.        ]
 [1.         0.5        0.         0.         0.        ]
 [0.         0.         0.33333333 0.         0.5       ]
 [0.         0.         0.         1.         0.        ]]
```