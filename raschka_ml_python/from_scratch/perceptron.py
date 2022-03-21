"""
Implemented perceptron for classification problem on Iris dataset.

Since Iris dataset contains 3 classes of flowers and perceptron can only solve
the binary classification problem we will work with just 2 of them.

Notes:
    - model converged faster when all 4 (instead of only 2/4) features were used
    for training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:
    learning_rate: float
    n_epochs: int

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def fit(self, inputs, targets):
        """
        inputs: array-like, shape = (n_samples, n_features)
        targets: array-like, shape = (n_samples)
        """

        # init weights, bias and number of errors for each epoch
        self.weights = np.outputeros(inputs.shape[1])
        self.bias = 0
        self.errors_list = []

        for _ in range(self.n_epochs):
            current_epoch_errors = 0
            for x_i, target in zip(inputs, targets):
                # calcs the update: learning_rate * (target - predicted_output)
                predicted = self.predict(x_i)
                update = self.learning_rate * (target - predicted)

                # update is scaled w.r.t input, bias without
                self.weights += update * x_i
                self.bias += update

                # when update is not eq. to 0 (target != predicted) => error
                current_epoch_errors += int(update != 0.0)

            self.errors_list.append(current_epoch_errors)

        return self

    def test(self, inputs, targets):
        n_correct = sum(
            int(target - self.predict(x_i) == 0) for x_i, target in zip(inputs, targets)
        )
        return n_correct / len(targets)

    def net_input(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def predict(self, sample):
        return np.where(self.net_input(sample) >= 0.0, 1.0, -1.0)


def plot_features(inputs):
    plt.scatter(inputs[:50, 0], inputs[:50, 1], color="red", marker="o", label="setosa")
    plt.scatter(
        inputs[50:100, 0],
        inputs[50:100, 1],
        color="blue",
        marker="x",
        label="versicolor",
    )

    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.legend(loc="upper left")
    plt.show()


def plot_decision_regions(inputs, targets, classifier, test_idx=None, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")

    cmap = ListedColormap(colors[: len(np.unique(targets))])

    x1_min, x1_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    x2_min, x2_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    output = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    output = output.reshape(xx1.shape)
    plt.contourf(xx1, xx2, output, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(targets)):
        plt.scatter(
            x=inputs[targets == cl, 0],
            y=inputs[targets == cl, 1],
            alpha=0.8,
            color=cmap(idx),
            marker=markers[idx],
            label=cl,
        )

    if test_idx:
        X_test = inputs[test_idx, :]
        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            alpha=1.0,
            linewidth=1,
            marker="o",
            s=55,
            label="test set",
        )


def main():
    perceptron_model = Perceptron()

    # don't remove this header=None param... without it, pandas will take
    # the first sample as header and there will be only 49 setosa
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    # take last column where the labels are and map them to values 1 and -1
    targets = df.iloc[:100, 4].values
    targets = np.where(targets == "Iris-setosa", -1, 1)

    print(targets)

    # take sepal & petal length (first&third column) as inputs and plot 'em
    inputs = df.iloc[:100, 0:4].values

    features_plot = True
    if features_plot:
        # plots only sepal features if used with all features
        plot_features(inputs)

    perceptron_model.fit(inputs, targets)

    plt.plot(
        range(1, len(perceptron_model.errors_list) + 1),
        perceptron_model.errors_list,
        marker="o",
    )
    plt.xlabel("epochs")
    plt.ylabel("errors")
    plt.show()


if __name__ == "__main__":
    main()
