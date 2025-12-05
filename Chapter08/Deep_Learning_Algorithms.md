# Chapter 8

# 딥러닝 (Deep Learning)

## 경사 하강법 정의 (Defining Gradient Descent)

```python
def adjust_position(gradient):
    while gradient != 0:
        if gradient < 0:
            print("Move right")
            # here would be your logic to move right
        elif gradient > 0:
            print("Move left")
            # here would be your logic to move left
```

---

## 활성화 함수 (Activation functions)

```python
def sigmoidFunction(z):
      return 1/ (1+np.exp(-z))
```

---

## ReLu

```python
def relu(x):
    if x < 0:
        return 0
    else:
        return x

def leaky_relu(x, beta=0.01):
    if x < 0:
        return beta * x
    else:
        return x
```

---

## 하이퍼볼릭 탄젠트 (Hyperbolic tangent)

```python
import numpy as np

def tanh(x):
    numerator = 1 - np.exp(-2 * x)
    denominator = 1 + np.exp(-2 * x)
    return numerator / denominator
```

---

## Softmax

```python
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
```

---

## Keras 모델 정의 (Defining a Keras model)

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(10, activation='softmax'),
])
```

---

## Functional API 방식의 Keras 모델 정의 (Functional API way of defining a Keras model)

```python
# Ensure TensorFlow 2.x is being used
%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Using the Functional API
inputs = tf.keras.Input(shape=(28, 28))  # Adjusted for MNIST
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(512, activation='relu', name='d1')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='d2')(x)  # 10 classes for 10 digits
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# One-hot encode the labels
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, 10)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

# Define the learning process
optimizer = tf.keras.optimizers.RMSprop()
loss = 'categorical_crossentropy'
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
history = model.fit(train_images, train_labels_one_hot, epochs=10, validation_data=(test_images, test_labels_one_hot))
```

**결과:**
```
Colab only includes TensorFlow 2.x; %tensorflow_version has no effect.
Epoch 1/10
1875/1875 [==============================] - 26s 13ms/step - loss: 0.2204 - accuracy: 0.9339 - val_loss: 0.1068 - val_accuracy: 0.9681
Epoch 2/10
1875/1875 [==============================] - 18s 10ms/step - loss: 0.1030 - accuracy: 0.9691 - val_loss: 0.0907 - val_accuracy: 0.9742
Epoch 3/10
1875/1875 [==============================] - 18s 9ms/step - loss: 0.0788 - accuracy: 0.9771 - val_loss: 0.0838 - val_accuracy: 0.9770
Epoch 4/10
1875/1875 [==============================] - 19s 10ms/step - loss: 0.0639 - accuracy: 0.9808 - val_loss: 0.0730 - val_accuracy: 0.9801
Epoch 5/10
1875/1875 [==============================] - 18s 10ms/step - loss: 0.0537 - accuracy: 0.9841 - val_loss: 0.0719 - val_accuracy: 0.9813
Epoch 6/10
1875/1875 [==============================] - 19s 10ms/step - loss: 0.0456 - accuracy: 0.9866 - val_loss: 0.0745 - val_accuracy: 0.9814
Epoch 7/10
1875/1875 [==============================] - 18s 10ms/step - loss: 0.0408 - accuracy: 0.9881 - val_loss: 0.0654 - val_accuracy: 0.9835
Epoch 8/10
1875/1875 [==============================] - 20s 11ms/step - loss: 0.0375 - accuracy: 0.9894 - val_loss: 0.0646 - val_accuracy: 0.9836
Epoch 9/10
1875/1875 [==============================] - 18s 10ms/step - loss: 0.0301 - accuracy: 0.9911 - val_loss: 0.0735 - val_accuracy: 0.9828
Epoch 10/10
1875/1875 [==============================] - 21s 11ms/step - loss: 0.0274 - accuracy: 0.9921 - val_loss: 0.0734 - val_accuracy: 0.9831
```

---

## 텐서 수학 이해하기 (Understanding Tensor Mathematics)

```python
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)
```

**결과:**
```
Define constant tensors
a = 2
b = 3
```

---

## 연산 실행 (Running operations)

```python
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)
```

**결과:**
```
Running operations, without tf.Session
a + b = 5
a * b = 6
```

---

## 텐서 덧셈

```python
c = a + b
print("a + b = %s" % c)
```

**결과:**
```
a + b = tf.Tensor(5, shape=(), dtype=int32)
```

---

## 텐서 곱셈

```python
d = a*b
print("a * b = %s" % d)
```

**결과:**
```
a * b = tf.Tensor(6, shape=(), dtype=int32)
```