# Chapter 09

# 자연어 처리 (Natural Language Processing)

## 토큰화 (Tokenization)

```python
import nltk
nltk.download('punkt')
```

**결과:**
```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
True
```

---

## 단어 토큰화 (Word Tokenization)

```python
from nltk.tokenize import word_tokenize
corpus = 'This is a book about algorithms.'

tokens = word_tokenize(corpus)
print(tokens)
```

**결과:**
```
['This', 'is', 'a', 'book', 'about', 'algorithms', '.']
```

---

## 문장 토큰화 (Sentence Tokenization)

```python
from nltk.tokenize import sent_tokenize
corpus = 'This is a book about algorithms. It covers various topics in depth.'
```

```python
sentences = sent_tokenize(corpus)
print(sentences)
```

**결과:**
```
['This is a book about algorithms.', 'It covers various topics in depth.']
```

---

## 단락 토큰화 함수 (Paragraph Tokenization Function)

```python
def tokenize_paragraphs(text):
    # Split by two newline characters
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p]
```

---

## Python을 사용한 데이터 정제 (Cleaning data using Python)

```python
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Make sure to download the NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
```

**결과:**
```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
```

---

## 텍스트 정제 함수 (Text Cleaning Function)

```python
def clean_text(text):
    """
    Cleans input text by converting case, removing punctuation, numbers, white spaces, stop words and stemming
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove white spaces
    text = text.strip()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_text)

    # Stemming
    ps = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_text = [ps.stem(word) for word in tokens]
    text = ' '.join(stemmed_text)

    return text
```

---

## 텍스트 정제 함수 테스트

```python
corpus="7- Today, Ottawa is becoming cold again "
clean_text(corpus)
```

**결과:**
```
'today ottawa becom cold'
```

---

## 문서 행렬 (Document Matrix) - CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

# Define a list of documents
documents = ["Machine Learning is useful", "Machine Learning is fun", "Machine Learning is AI"]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the documents into a TDM
tdm = vectorizer.fit_transform(documents)

# Print the TDM
print(tdm.toarray())
```

**결과:**
```
[[0 0 1 1 1 1]
 [0 1 1 1 1 0]
 [1 0 1 1 1 0]]
```

---

## TF-IDF 벡터화 (TF-IDF Vectorization)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a list of documents
documents = ["Machine Learning enables learning", "Machine Learning is fun", "Machine Learning is useful"]

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Loop over the feature names and print the TF-IDF score for each term
for i, term in enumerate(feature_names):
    tfidf = tfidf_matrix[:, i].toarray().flatten()
    print(f"{term}: {tfidf}")
```

**결과:**
```
enables: [0.60366655 0.         0.        ]
fun: [0.         0.66283998 0.        ]
is: [0.         0.50410689 0.50410689]
learning: [0.71307037 0.39148397 0.39148397]
machine: [0.35653519 0.39148397 0.39148397]
useful: [0.         0.         0.66283998]
```

---

## Word2Vec을 이용한 단어 임베딩 (Implementing word embedding with Word2Vec)

```python
import gensim

# Define a text corpus
corpus = [['apple', 'banana', 'orange', 'pear'],
          ['car', 'bus', 'train', 'plane'],
          ['dog', 'cat', 'fox', 'fish']]

# Train a word2vec model on the corpus
model = gensim.models.Word2Vec(corpus, window=5, min_count=1, workers=4)
print(model.wv.similarity('car', 'train'))
```

**결과:**
```
-0.057745814
```

```python
print(model.wv.similarity('car', 'apple'))
```

**결과:**
```
0.11117952
```

---

## 사례 연구: 레스토랑 리뷰 감성 분석 (Case study: Restaurant review sentiment analysis)

```python
import nltk
nltk.download('stopwords')
```

**결과:**
```
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
```

---

## 데이터 로드

```python
import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
url = 'https://storage.googleapis.com/neurals/data/2023/Restaurant_Reviews.tsv'
dataset = pd.read_csv(url, delimiter='\t', quoting=3)
dataset.head()
```

**결과:**
```
Review	Liked
0	Wow... Loved this place.	1
1	Crust is not good.	0
2	Not tasty and the texture was just nasty.	0
3	Stopped by during the late May bank holiday of...	1
4	The selection on the menu was great and so wer...	1
```

---

## 텍스트 정제 및 코퍼스 생성

```python
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [
        ps.stem(word) for word in text
        if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

corpus = [clean_text(review) for review in dataset['Review']]
```

---

## 모델 학습 및 평가

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Initialize the CountVectorizer and transform the corpus
vectorizer = CountVectorizer(max_features=1500)
X = vectorizer.fit_transform(corpus).toarray()

# Get the target labels
y = dataset.iloc[:, 1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Initialize and train the Gaussian Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

**결과:**
```
[[55 42]
 [12 91]]
```