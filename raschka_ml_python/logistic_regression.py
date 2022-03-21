"""
The simple linear activation function in adaline will now be replaced with
sigmoid function, which models the class probabilty by taking a real number and mapping
it to values between 0, 1. Sigmoid(z) = 1 / 1 + e^-z

Logistic Regression model uses a log-likelihood cost function which is to be minimized
through gradient descent.

Log-likelihood for one sample: J(phi_z, y;w) = -ylog(phi_z) - (1 - y)log(1 - phi_z)
which for class y = 1 will calculate according to first term -log(phi_z) and for
class y = 0 will calculate according to the second term -log(1 - phi_z)
=> punishing wrong prediction with increasingly large cost
"""


from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

from from_scratch.perceptron import plot_decision_regions


""" def sigmoid(z: float):
    return 1.0 / (1.0 + np.exp(-z))


# plot sigmoid in range -7,7
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z, "r-")
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel("z")
plt.ylabel("$\phi (z)$")
plt.show() """

# prepare data
iris_dataset = load_iris()
X = iris_dataset.data[:, [0, 2]]
y = iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# logistic regression model
lr_model = LogisticRegression(C=1000.0, random_state=0)
lr_model.fit(X_train_std, y_train)

plot_decision_regions(
    X_combined_std, y_combined, classifier=lr_model, test_idx=range(105, 150)
)
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc="upper left")
plt.show()

predicted = lr_model.predict(X_test_std)
print(f"Linear Regression model accuracy: {accuracy_score(y_test, predicted)}")
