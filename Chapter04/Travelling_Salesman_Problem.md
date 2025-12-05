# Chapter 4

# 외판원 문제  

## 1- 무차별 대입 전략  

```python
import random
from itertools import permutations
import matplotlib.pyplot as plt
from collections import Counter
from time import time
```

---

## 거리 및 경로 계산  

```python
aCity = complex

def distance_points(first, second):
    return abs(first - second)

def distance_tour(aTour):
    return sum(distance_points(aTour[i - 1], aTour[i])
               for i in range(len(aTour))
               )

def generate_cities(number_of_cities):
    seed = 111
    width = 500
    height = 300
    random.seed((number_of_cities, seed))
    return frozenset(aCity(random.randint(1, width), random.randint(1, height))
                     for c in range(number_of_cities))
```

---

## 무차별 대입 알고리즘  

```python
def brute_force(cities):
    return shortest_tour(permutations(cities))

def shortest_tour(tours):
    return min(tours, key=distance_tour)
```

---

## 시각화  

```python
def visualize_tour(tour, style='bo-'):
    if len(tour) > 1000:
        plt.figure(figsize=(15, 10))
    start = tour[0:1]
    visualize_segment(tour + start, style)
    visualize_segment(start, 'rD')

def visualize_segment(segment, style='bo-'):
    plt.plot([X(c) for c in segment], [Y(c) for c in segment], style, clip_on=False)
    plt.axis('scaled')
    plt.axis('off')

def X(city):
    return city.real

def Y(city):
    return city.imag
```

---

## TSP 함수 

```python
def tsp(algorithm, cities):
    t0 = time()
    tour = algorithm(cities)
    t1 = time()
    # Every city appears exactly once in tour
    assert Counter(tour) == Counter(cities)
    visualize_tour(tour)
    print("{}: {} cities => tour length {:.0f} (in {:.3f} sec)".format(
        name(algorithm), len(tour), distance_tour(tour), t1-t0))

def name(algorithm):
    return algorithm.__name__.replace('_tsp', '')
```

---

## 실행  

```python
tsp(brute_force, generate_cities(10))
```

---

## 2- 탐욕 알고리즘  


```python
# Greedy Algorithm for TSP
def greedy_algorithm(cities, start=None):
    city_ = start or first(cities)
    tour = [city_]
    unvisited = set(cities - {city_})
    while unvisited:
        city_ = nearest_neighbor(city_, unvisited)
        tour.append(city_)
        unvisited.remove(city_)
    return tour

def first(collection):
    return next(iter(collection))

def nearest_neighbor(city_a, cities):
    return min(cities, key=lambda city_: distance_points(city_, city_a))

# Now, let's use greedy_algorithm to create a tour for 2,000 cities
tsp(greedy_algorithm, generate_cities(2000))
```