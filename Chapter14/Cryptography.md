# Chapter 14

# 암호학 

## 시저 암호  

```python
rotation = 3
P = 'CALM'; C=''
for letter in P:
    C = C+ (chr(ord(letter) + rotation))
print(C)
```

**결과:**
```
FDOP
```

---

## ROT 13

```python
rotation = 13
P = 'CALM'; C=''
for letter in P:
    C = C+ (chr(ord(letter) + rotation))
print(C)
```

**결과:**
```
PNYZ
```

---

## MD5 해시 이해하기  

```python
import hashlib

def generate_md5_hash(input_string):
    # Create a new md5 hash object
    md5_hash = hashlib.md5()

    # Encode the input string to bytes and hash it
    md5_hash.update(input_string.encode())

    # Return the hexadecimal representation of the hash
    return md5_hash.hexdigest()

def verify_md5_hash(input_string, correct_hash):
    # Generate md5 hash for the input_string
    computed_hash = generate_md5_hash(input_string)

    # Compare the computed hash with the provided hash
    return computed_hash == correct_hash

# Test
input_string = "Hello, World!"
hash_value = generate_md5_hash(input_string)
print(f"Generated hash: {hash_value}")

correct_hash = hash_value
print(verify_md5_hash(input_string, correct_hash))  # This should return True
```

**결과:**
```
Generated hash: 65a8e27d8879283831b664bd8b7f0ad4
True
```

---

## 보안 해싱 알고리즘 이해하기  

```python
import hashlib
salt = "qIo0foX5"
password = "myPassword"
salted_password = salt + password
sha512_hash = hashlib.sha512()
sha512_hash.update(salted_password.encode())
myHash = sha512_hash.hexdigest()
myHash
```

**결과:**
```
'2e367911b87b12f73b135b1a4af9fac193a8064d3c0a52e34b3a52a5422beed2b6276eabf95abe728f91ba61ef93175e5bac9a643b54967363ffab0b35133563'
```

---

## 대칭 암호화 코딩 

```python
import hashlib
sha256_hash = hashlib.sha256()
message = "Ottawa is really cold".encode()
sha256_hash.update(message)
print(sha256_hash.hexdigest())
```

**결과:**
```
b6ee63a201c4505f1f50ff92b7fe9d9e881b57292c00a3244008b76d0e026161
```

---

## MITM 공격 방지 방법 

```python
from xmlrpc.client import SafeTransport, ServerProxy
import ssl
```

---

## 데이터 및 모델 암호화  

```python
# 1. Install and import required libraries
!pip install cryptography

import pickle
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from cryptography.fernet import Fernet

# 2. Train a simple model using the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train, y_train)

# 3. Define the names of the files that will store the model
filename_source = "unencrypted_model.pkl"
filename_destination = "decrypted_model.pkl"
filename_sec = "encrypted_model.pkl"

# 4. Store the trained model in a file
dump(model, filename_source)

# 5. Define functions for encryption and decryption
def write_key():
    key = Fernet.generate_key()
    with open("key.key", "wb") as key_file:
        key_file.write(key)

def load_key():
    return open("key.key", "rb").read()

def encrypt(filename, key):
    f = Fernet(key)
    with open(filename, "rb") as file:
        file_data = file.read()
    encrypted_data = f.encrypt(file_data)
    with open(filename_sec, "wb") as file:
        file.write(encrypted_data)

def decrypt(filename, key):
    f = Fernet(key)
    with open(filename, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = f.decrypt(encrypted_data)
    with open(filename_destination, "wb") as file:
        file.write(decrypted_data)

# 6. Use the functions to encrypt the model, then decrypt it
write_key()
key = load_key()
encrypt(filename_source, key)
decrypt(filename_sec, key)

# 7. Load the decrypted model and make predictions
loaded_model = load(filename_destination)
result = loaded_model.score(X_test, y_test)
print(result)
```

**결과:**
```
Requirement already satisfied: cryptography in /usr/local/lib/python3.10/dist-packages (41.0.3)
Requirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.10/dist-packages (from cryptography) (1.15.1)
Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.12->cryptography) (2.21)
1.0
```

---

## 현재 작업 디렉토리 확인

```python
import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_directory)
```

**결과:**
```
Current Working Directory: /content
```

---

## 파일 목록 확인

```python
!ls
```

**결과:**
```
decrypted_model.pkl  key.key	  unencrypted_model.pkl
encrypted_model.pkl  sample_data
```