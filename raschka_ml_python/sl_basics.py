from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from from_scratch.perceptron import plot_decision_regions

# loading and splitting iris dataset
iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, [0, 2]]
y = iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# perceptron model
perceptron_model = Perceptron(max_iter=40, eta0=0.01, random_state=1)
perceptron_model.fit(X_train_std, y_train)

predicted = perceptron_model.predict(X_test_std)
print(f"Num of misclassified samples: {(predicted != y_test).sum()}")

print(f"Perceptron model accuracy: {accuracy_score(y_test, predicted)}")

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(
    inputs=X_combined_std,
    targets=y_combined,
    classifier=perceptron_model,
    test_idx=range(105, 150),
)
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc="upper left")
plt.show()
