# Chapter 10

# 순차 모델 이해하기 (Understanding Sequential Models)

## 데이터 로드 및 전처리

```python
import numpy as np
import tensorflow as tf

# Load the dataset
vocab_size = 50000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# Pad the sequences
max_review_length = 500
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length)

# Define reverse word index to convert integer sequences back to words
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(padded_sequence):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in padded_sequence])

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=50, input_length=max_review_length))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.LSTM(units=32))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
```

**결과:**
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17464789/17464789 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
1641221/1641221 [==============================] - 0s 0us/step
```

---

## 모델 학습

```python
# Train the model
history = model.fit(x_train, y_train, batch_size=256, epochs=3, validation_split=0.2, verbose=1)
model.summary
```

**결과:**
```
Epoch 1/3
79/79 [==============================] - 66s 802ms/step - loss: 0.6009 - accuracy: 0.6799 - val_loss: 0.4065 - val_accuracy: 0.8388
Epoch 2/3
79/79 [==============================] - 62s 781ms/step - loss: 0.2884 - accuracy: 0.8923 - val_loss: 0.2857 - val_accuracy: 0.8840
Epoch 3/3
79/79 [==============================] - 61s 776ms/step - loss: 0.1661 - accuracy: 0.9460 - val_loss: 0.2995 - val_accuracy: 0.8880

<bound method Model.summary of <keras.src.engine.sequential.Sequential object at 0x794f28b31ba0>>
```

---

## 예측 및 오분류 리뷰 확인

```python
# Predictions
predicted_probs = model.predict(x_test)
predicted_classes_reshaped = (predicted_probs > 0.5).astype("int32").reshape(-1)
incorrect = np.nonzero(predicted_classes_reshaped != y_test)[0]

# Display some incorrect predictions
class_names = ["Negative", "Positive"]
for j, incorrect_index in enumerate(incorrect[0:20]):
    predicted = class_names[predicted_classes_reshaped[incorrect_index]]
    actual = class_names[y_test[incorrect_index]]
    human_readable_review = decode_review(x_test[incorrect_index])
    print(f"Incorrectly classified Test Review [{j+1}]")
    print(f"Test Review #{incorrect_index}: Predicted [{predicted}] Actual [{actual}]")
    print(f"Test Review Text: {human_readable_review.replace('<PAD> ', '')}\n")
```