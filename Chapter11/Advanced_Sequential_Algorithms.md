# Chapter 11

# 고급 순차 모델링 알고리즘 (Advanced Sequential Modeling Algorithms)

## Part1- 오토인코더 코딩 (Coding Autoencoders)

### 필요한 라이브러리 임포트 (Import Necessary Libraries)

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

---

### MNIST 데이터 로드 (Load the MNIST Data)

```python
# Load dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize data to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0
```

---

### 모델 정의 (Define the Model)

```python
# Define the autoencoder model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28))
])
```

---

### 모델 컴파일 (Compile the Model)

```python
model.compile(loss='binary_crossentropy', optimizer='adam')
```

---

### 모델 학습 (Train the Model)

```python
model.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))
```

---

### 예측 (Prediction)

```python
# For an autoencoder, the encoder and decoder parts are usually separate.
# Here, the entire autoencoder is used for encoding and decoding.
encoded_data = model.predict(x_test)
decoded_data = model.predict(encoded_data)
```

---

### 시각화 (Visualization)

```python
# Display original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_data[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
```

---

## Part2- 셀프 어텐션 (Self Attention)

### 라이브러리 임포트 (Importing necessary libraries)

```python
import numpy as np
```

---

### 셀프 어텐션 함수 정의 (Defining the self-attention function)

```python
def self_attention(Q, K, V):
    """
    Q: Query matrix
    K: Key matrix
    V: Value matrix
    """

    # Calculate the attention weights
    attention_weights = np.matmul(Q, K.T)

    # Apply the softmax to get probabilities
    attention_probs = np.exp(attention_weights) / np.sum(np.exp(attention_weights), axis=1, keepdims=True)

    # Multiply the probabilities with the value matrix to get the output
    output = np.matmul(attention_probs, V)

    return output
```

---

### 예제 사용 (Example Usage)

### 행렬 초기화 (Initialize matrices)

```python
Q = np.array([[1, 0, 1], [0, 2, 0], [1, 1, 0]])  # Example Query
K = np.array([[1, 0, 1], [0, 2, 0], [1, 1, 0]])  # Key matrix
V = np.array([[0, 2, 0], [1, 0, 1], [0, 1, 2]])  # Value matrix
```

---

### 셀프 어텐션 함수를 사용하여 출력 계산 (Compute the output using the self_attention function)

```python
output = self_attention(Q, K, V)
```

---

### 결과 출력 (Display the result)

```python
print(output)
```