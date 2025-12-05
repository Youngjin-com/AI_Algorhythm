# Chapter 8

# 샴 네트워크 

## 임포트  

```python
import random
import numpy as np
import tensorflow as tf
```

---

## 데이터 준비 함수 

```python
def prepareData(inputs: np.ndarray, labels: np.ndarray):
      classesNumbers = 10
      digitalIdx = [np.where(labels == i)[0] for i in range(classesNumbers)]
      pairs = list()
      labels = list()
      n = min([len(digitalIdx[d]) for d in range(classesNumbers)]) - 1
      for d in range(classesNumbers):
        for i in range(n):
            z1, z2 = digitalIdx[d][i], digitalIdx[d][i + 1]
            pairs += [[inputs[z1], inputs[z2]]]
            inc = random.randrange(1, classesNumbers)
            dn = (d + inc) % classesNumbers
            z1, z2 = digitalIdx[d][i], digitalIdx[dn][i]
            pairs += [[inputs[z1], inputs[z2]]]
            labels += [1, 0]
      return np.array(pairs), np.array(labels, dtype=np.float32)
```

---

## 템플릿 생성 함수  

```python
def createTemplate():
      return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation='relu'),
        ])
```

---

## MNIST 데이터셋 로드

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
```

**결과:**
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 1s 0us/step
```

---

## 훈련 및 테스트 데이터셋 생성

```python
train_pairs, tr_labels = prepareData(x_train, y_train)
test_pairs, test_labels = prepareData(x_test, y_test)
```

---

## 베이스 네트워크 생성

```python
base_network = createTemplate()
```

---

## 샴 네트워크의 두 브랜치 생성

```python
# Create first half of the siamese system
input_a = tf.keras.layers.Input(shape=input_shape)
# Note how we reuse the base_network in both halfs
enconder1 = base_network(input_a)
# Create the second half of the siamese system
input_b = tf.keras.layers.Input(shape=input_shape)
enconder2 = base_network(input_b)
```

---

## 유사도 측정 레이어 생성

```python
distance = tf.keras.layers.Lambda(
    lambda embeddings: tf.keras.backend.abs(embeddings[0] - embeddings[1])) \
    ([enconder1, enconder2])
```

---

## 최종 출력 레이어 생성

```python
measureOfSimilarity = tf.keras.layers.Dense(1, activation='sigmoid') (distance)
```

---

## 모델 빌드 및 훈련

```python
# Build the model
model = tf.keras.models.Model([input_a, input_b], measureOfSimilarity)
# Train
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

model.fit([train_pairs[:, 0], train_pairs[:, 1]], tr_labels,
          batch_size=128,epochs=10,validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels))
```

**결과:**
```
Epoch 1/10
847/847 [==============================] - 17s 16ms/step - loss: 0.3535 - accuracy: 0.8475 - val_loss: 0.2727 - val_accuracy: 0.9097
Epoch 2/10
847/847 [==============================] - 9s 11ms/step - loss: 0.1845 - accuracy: 0.9314 - val_loss: 0.1970 - val_accuracy: 0.9374
Epoch 3/10
847/847 [==============================] - 9s 10ms/step - loss: 0.1251 - accuracy: 0.9552 - val_loss: 0.1395 - val_accuracy: 0.9570
Epoch 4/10
847/847 [==============================] - 8s 9ms/step - loss: 0.0951 - accuracy: 0.9661 - val_loss: 0.1184 - val_accuracy: 0.9644
Epoch 5/10
847/847 [==============================] - 9s 11ms/step - loss: 0.0781 - accuracy: 0.9725 - val_loss: 0.1033 - val_accuracy: 0.9698
Epoch 6/10
847/847 [==============================] - 9s 10ms/step - loss: 0.0656 - accuracy: 0.9772 - val_loss: 0.0954 - val_accuracy: 0.9718
Epoch 7/10
847/847 [==============================] - 8s 9ms/step - loss: 0.0563 - accuracy: 0.9802 - val_loss: 0.0884 - val_accuracy: 0.9730
Epoch 8/10
847/847 [==============================] - 9s 10ms/step - loss: 0.0505 - accuracy: 0.9824 - val_loss: 0.0862 - val_accuracy: 0.9741
Epoch 9/10
847/847 [==============================] - 9s 10ms/step - loss: 0.0443 - accuracy: 0.9846 - val_loss: 0.0911 - val_accuracy: 0.9723
Epoch 10/10
847/847 [==============================] - 8s 9ms/step - loss: 0.0416 - accuracy: 0.9855 - val_loss: 0.0793 - val_accuracy: 0.9769
<keras.src.callbacks.History at 0x7cd16686d8a0>
```