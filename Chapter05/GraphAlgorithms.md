# Chapter 5

# 그래프 알고리즘 (Graph Algorithms)

## 그래프 구조 및 유형 (Graph mechanics and types)

```python
import networkx as nx
graph = nx.Graph()
graph.add_node("Mike")
graph.add_nodes_from(["Amine", "Wassim", "Nick"])
graph.add_edge("Mike", "Amine")
print(graph.nodes())
print(graph.edges())
```

**결과:**
```
['Mike', 'Amine', 'Wassim', 'Nick']
[('Mike', 'Amine')]
```

```python
graph.add_edge("Amine", "Imran")
print(graph.edges())
```

**결과:**
```
[('Mike', 'Amine'), ('Amine', 'Imran')]
```

---

## 중심성 지표 계산 (Calculating centrality metrics)

```python
import networkx as nx
import matplotlib.pyplot as plt
vertices = range(1, 10)
edges = [(7, 2), (2, 3), (7, 4), (4, 5), (7, 3), (7, 5), (1, 6), (1, 7), (2, 8), (2, 9)]
```

---

## 그래프 생성 (Crafting the Graph)

```python
graph = nx.Graph()
graph.add_nodes_from(vertices)
graph.add_edges_from(edges)
```

---

## 시각화 (Painting a picture)

```python
nx.draw(graph, with_labels=True, node_color='y', node_size=800)
plt.show()
```

**결과:**
```
No description has been provided for this image
```

```python
print("Degree Centrality:", nx.degree_centrality(graph))
```

**결과:**
```
Degree Centrality: {1: 0.25, 2: 0.5, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.125, 7: 0.625, 8: 0.125, 9: 0.125}
```

```python
print("Betweenness Centrality:", nx.betweenness_centrality(graph))
```

**결과:**
```
Betweenness Centrality: {1: 0.25, 2: 0.46428571428571425, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.7142857142857142, 8: 0.0, 9: 0.0}
```

```python
print("Closeness Centrality:", nx.closeness_centrality(graph))
```

**결과:**
```
Closeness Centrality: {1: 0.5, 2: 0.6153846153846154, 3: 0.5333333333333333, 4: 0.47058823529411764, 5: 0.47058823529411764, 6: 0.34782608695652173, 7: 0.7272727272727273, 8: 0.4, 9: 0.4}
```

```python
eigenvector_centrality = nx.eigenvector_centrality(graph)
sorted_centrality = sorted((vertex, '{:0.2f}'.format(centrality_val))
                           for vertex, centrality_val in eigenvector_centrality.items())
print("Eigenvector Centrality:", sorted_centrality)
```

**결과:**
```
Eigenvector Centrality: [(1, '0.24'), (2, '0.45'), (3, '0.36'), (4, '0.32'), (5, '0.32'), (6, '0.08'), (7, '0.59'), (8, '0.16'), (9, '0.16')]
```

---

## BFS

```python
graph={ 'Amin'   : {'Wasim', 'Nick', 'Mike'},
         'Wasim' : {'Imran', 'Amin'},
         'Imran' : {'Wasim','Faras'},
         'Faras' : {'Imran'},
         'Mike'  : {'Amin'},
         'Nick' :  {'Amin'}}
```

---

## BFS 알고리즘 구현 (BFS Algorithm Implementation)

```python
def bfs(graph, start):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            neighbours = graph[node]

            # Only add neighbors to the queue that have not been visited yet
            unvisited_neighbours = [neighbour for neighbour in neighbours if neighbour not in visited]
            queue.extend(unvisited_neighbours)

    return visited

# Test


start_node = 'Amin'
print(bfs(graph, start_node))
```

**결과:**
```
{'Nick', 'Faras', 'Mike', 'Amin', 'Imran', 'Wasim'}
```

---

## DFS

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

graph={ 'Amin' : {'Wasim', 'Nick', 'Mike'},
         'Wasim' : {'Imran', 'Amin'},
         'Imran' : {'Wasim','Faras'},
         'Faras' : {'Imran'},
         'Mike'  :{'Amin'},
         'Nick'  :{'Amin'}}
```

---

## 사례 연구: SNA를 활용한 사기 탐지 (Case study: fraud detection using SNA)

```python
import networkx as nx
import matplotlib.pyplot as plt

vertices = range(1,10)
edges= [(7,2), (2,3), (7,4), (4,5), (7,3), (7,5), (1,6),(1,7),(2,8),(2,9)]

graph = nx.Graph()

graph.add_nodes_from(vertices)
graph.add_edges_from(edges)
positions = nx.spring_layout(graph)

# Drawing the nodes with green color
nx.draw_networkx_nodes(graph, positions,
                       nodelist=[1, 4, 3, 8, 9],
                       node_color='g',
                       node_size=1300)

# No need to repeat the above call

labels = {1: '1 NF', 2: '2 F', 3: '3 NF', 4: '4 NF', 5: '5 F', 6: '6 F', 7: '7 F', 8: '8 NF', 9: '9 NF'}
nx.draw_networkx_labels(graph, positions, labels, font_size=16)

# Drawing the edges
nx.draw_networkx_edges(graph, positions, edges, width=3, alpha=0.5, edge_color='b')

plt.show()  # Display the graph
```