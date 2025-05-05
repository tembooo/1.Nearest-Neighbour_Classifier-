# 📊 k-Nearest Neighbour (k-NN) Classifier - 9D Dataset Project

## 📘 Task Description

Implement nearest-neighbour classifier in Matlab or Python. The function interface should be something like the following:

```python
testclass = knn(trainclass, traindata, testdata, k)
```

Where `traindata` and `trainclass` are the training data set and the corresponding class labels, `testdata` contains the test set, `k` is the number of neighbours used in the classification and the function returns a vector `testclass` containing the class labels for the test set. Use the Euclidean distance as the distance metric.

Experiment with the provided 9D data set. Divide the data into training and test sets so that 2/3 of the data (466 samples from the beginning of the data) is used for training and the rest for testing. **Do not shuffle the data** (for reproducibility of results). Test different values for `k` from 1 to 8.

For solving the two-class classification problem, the **best value for `k` is `3`**. ✅  
If several values of `k` produce best results, specify the smallest.

### 📎 Additional files:
- CSV
- MAT

### 💡 Hints:

1. The algorithm for kNN classification can be described as follows:
   - Find `k` nearest neighbours from the training set for a sample to be classified.
   - Classify the sample to the class which has the most training samples among the `k` nearest neighbours.

2. In the case that two or more classes have an equal highest number of samples among the `k` nearest neighbours, the value of `k` can be decreased until unambiguous classification is possible.
## 📘 Task Description

Implement nearest-neighbour classifier in Matlab or Python. The function interface should be something like the following:

```python
testclass = knn(trainclass, traindata, testdata, k)
```

Where `traindata` and `trainclass` are the training data set and the corresponding class labels, `testdata` contains the test set, `k` is the number of neighbours used in the classification and the function returns a vector `testclass` containing the class labels for the test set. Use the Euclidean distance as the distance metric.

Experiment with the provided 9D data set. Divide the data into training and test sets so that 2/3 of the data (466 samples from the beginning of the data) is used for training and the rest for testing. **Do not shuffle the data** (for reproducibility of results). Test different values for `k` from 1 to 8.

For solving the two-class classification problem, the **best value for `k` is `3`**. ✅  
If several values of `k` produce best results, specify the smallest.

### 📎 Additional files:
- CSV
- MAT

### 💡 Hints:

1. The algorithm for kNN classification can be described as follows:
   - Find `k` nearest neighbours from the training set for a sample to be classified.
   - Classify the sample to the class which has the most training samples among the `k` nearest neighbours.

2. In the case that two or more classes have an equal highest number of samples among the `k` nearest neighbours, the value of `k` can be decreased until unambiguous classification is possible.





Hi there! 👋  
In this repository, I implemented a basic **k-Nearest Neighbour (k-NN)** classifier from scratch in Python using **Euclidean distance**.  
We apply it to a **9-dimensional dataset** to perform two-class classification.

---

## 🧠 Step One: Load the Dataset and Split It

👉 Download the dataset file: `t031.csv`  
We use the **first 466 samples** (2/3 of the data) for training and the rest for testing — **no shuffling** (for reproducibility).

```python
import pandas as pd
import numpy as np

file_path = 't031.csv'
data = pd.read_csv(file_path)

features = data.iloc[:, :9].values
labels = data.iloc[:, -1].values

traindata = features[:466, :]
trainclass = labels[:466]
testdata = features[466:, :]
testclass_actual = labels[466:]
from collections import defaultdict

def knn_classify(test_sample, train_data, train_labels, max_k):
    dists = np.linalg.norm(train_data - test_sample, axis=1)
    sorted_indices = np.argsort(dists)
    sorted_labels = train_labels[sorted_indices]
    counts = defaultdict(int)
    results = np.full(max_k, sorted_labels[0])  # Default to nearest neighbor
    for i in range(1, max_k + 1):
        counts[sorted_labels[i - 1]] += 1
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_counts) > 1 and sorted_counts[0][1] == sorted_counts[1][1]:
            top_label = results[i - 2]
        else:
            top_label = sorted_counts[0][0]
        results[i - 1] = top_label
    return results
from collections import defaultdict

def knn_classify(test_sample, train_data, train_labels, max_k):
    dists = np.linalg.norm(train_data - test_sample, axis=1)
    sorted_indices = np.argsort(dists)
    sorted_labels = train_labels[sorted_indices]
    counts = defaultdict(int)
    results = np.full(max_k, sorted_labels[0])  # Default to nearest neighbor
    for i in range(1, max_k + 1):
        counts[sorted_labels[i - 1]] += 1
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_counts) > 1 and sorted_counts[0][1] == sorted_counts[1][1]:
            top_label = results[i - 2]
        else:
            top_label = sorted_counts[0][0]
        results[i - 1] = top_label
    return results
max_k = 8
testclass_pred = []
for test_sample in testdata:
    knn_results = knn_classify(test_sample, traindata, trainclass, max_k)
    testclass_pred.append(knn_results)
from sklearn.metrics import accuracy_score

accuracies = []
for k in range(1, max_k + 1):
    predicted_k = [result[k - 1] for result in testclass_pred]
    accuracy = accuracy_score(testclass_actual, predicted_k)
    accuracies.append(accuracy)

best_k = np.argmax(accuracies) + 1
print('Best k is =', best_k)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy for Different k Values')
plt.axvline(x=best_k, linestyle='--', color='r', label=f'Best k = {best_k}')
plt.legend()
plt.grid(True)
plt.show()

```
![image](https://github.com/user-attachments/assets/cd736b0a-4196-466a-ba20-3430e60f32d6)
