"""
Implemented ADAptive LInear NEuron

Diff:
    Introduced linear activation function with non-binary update propagation ->
    weights are updated through gradient descent.
    Moved from incremental to batch learning -> pass through the whole dataset and then update

    We define cost function J(w) = 1/2*sum(y_i - activation(net_input))**2 that tells us
    how the model performs and if we calculate the partial derivative w.r.t weights it can
    tell us the direction which we should update the weights (gradient) in order to minimize this function.
"""

from random import seed
from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import plot_decision_regions


class Adaline:
    learning_rate: float
    n_epochs: int
    shuffle: bool
    random_seed: int
    weights_init: bool = False

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 10,
        shuffle=True,
        random_seed=None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_seed = random_seed
        if random_seed:
            seed(random_seed)
        self.cost_list = []

    def fit_batch_gd(self, X, y):
        # the convergence scale and speed differs here in weight init
        # for zero init -> linear convergence, scale: 10^1
        # for random init -> exponential convergence, scale 10^3
        # true without the standardization step !

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_epochs):
            output = self.net_input(X)
            errors = y - output

            # update weights&bias
            # X.T.shape = (4, 100) so we can matmul with errors (100, 1)
            # result shape is (4, 1)... in numpy same as self.weights.shape
            # bias is just summarized errors scaled with learning_rate
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()

            # calc the cost function
            cost = (errors**2).sum() / 2.0
            self.cost_list.append(cost)

        return self

    def fit_sgd(self, X, y):
        self.init_weights(X.shape[1])

        for _ in range(self.n_epochs):
            if self.shuffle:
                self.shuffle_data(X, y)
            epoch_cost_list = [
                self.update_weights(x_i, target) for x_i, target in zip(X, y)
            ]
            avg_epoch_cost = sum(epoch_cost_list) / len(y)
            self.cost_list.append(avg_epoch_cost)

        return self

    def online_fit(self, X, y):
        if not self.weights_init:
            self.init_weights(X.shape[1])
        # if a batch of samples came -> update for each input
        if y.ravel().shape[0] > 1:
            samples_cost_list = [
                self.update_weights(x_i, target) for x_i, target in zip(X, y)
            ]

            self.cost_list.append(sum(samples_cost_list) / len(y))
        else:
            self.update_weights(X, y)

        return self

    def init_weights(self, shape):
        self.weights = np.zeros(shape)
        self.bias = 0
        self.weights_init = True

    def update_weights(self, x_i, target):
        output = self.net_input(x_i)
        error = target - output
        self.weights += self.learning_rate * x_i.dot(error)
        self.bias += self.learning_rate * error

        return (error**2) / 2

    def shuffle_data(self, X, y):
        # get random permutation of dataset -> array of shuffled indices
        shuffled = np.random.permutation(len(y))
        return X[shuffled], y[shuffled]

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        # for now just an identity function
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def main():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    train_X = df.iloc[:100, 0:4].values

    train_y = df.iloc[:100, 4].values
    train_y = np.where(train_y == "Iris-setosa", -1, 1)

    adaline_fast = Adaline().fit_batch_gd(train_X, train_y)
    adaline_slow = Adaline(learning_rate=0.0001).fit_batch_gd(train_X, train_y)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].plot(
        range(1, adaline_fast.n_epochs + 1),
        np.log10(adaline_fast.cost_list),
        marker="o",
    )
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("log-SSE (cost)")

    ax[1].plot(
        range(1, adaline_slow.n_epochs + 1),
        adaline_slow.cost_list,
        marker="o",
    )
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("SSE (cost)")

    plt.show()

    # now with feature scaling (standardization)
    # each feature column x_j update as: x_j = (x_j - mean(x_j)) / std(x_j)

    # example with only 2 features (sepal&petal length) to showcase the decision boundary
    train_X_std = np.copy(train_X)[:, [0, 2]]
    train_X_std[:, 0] = (train_X_std[:, 0] - train_X_std[:, 0].mean()) / train_X_std[
        :, 0
    ].std()

    train_X_std[:, 1] = (train_X_std[:, 1] - train_X_std[:, 1].mean()) / train_X_std[
        :, 1
    ].std()

    """ adaline_bgd = Adaline(learning_rate=0.001, n_epochs=15).fit_batch_gd(
        train_X_std, train_y
    )
    plot_decision_regions(train_X_std, train_y, classifier=adaline_bgd) """

    # example with standardization across each column (all 4 features)
    """ train_X_std = np.copy(train_X)
    for i, col in enumerate(train_X.T):
        train_X_std[:, i] = (train_X[:, i] - col.mean()) / col.std() """

    """ adaline_bgd = Adaline(learning_rate=0.001, n_epochs=15).fit_batch_gd(train_X_std, train_y)
    plt.plot(
        range(1, len(adaline_std.cost_list) + 1), adaline_std.cost_list, marker="o"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Sum-squared-error")
    plt.show() """

    # example of stochastic gd learning
    """ adaline_sgd = Adaline(learning_rate=0.01, n_epochs=10, random_seed=1).fit_sgd(
        train_X_std, train_y
    )
    plot_decision_regions(train_X_std, train_y, classifier=adaline_sgd)
    plt.show()
    plt.plot(
        range(1, len(adaline_sgd.cost_list) + 1), adaline_sgd.cost_list, marker="o"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Sum-squared-error")
    plt.show() """

    # example for online learning (every 5 seconds a batch of 5 datapoints arrives)
    # dynamic plot init and configuration

    plt.ion()
    fig, ax = plt.subplots()
    (lines,) = ax.plot([], [], marker="o")
    ax.set_autoscaley_on(True)
    ax.set_xlim(1, 20)
    plt.xticks(range(1, 20 + 1))
    plt.xlabel("Pass")
    plt.ylabel("Sum-squared-error")

    # shuffle data before start
    shuffled = np.random.permutation(len(train_y))
    train_X_std, train_y = train_X_std[shuffled], train_y[shuffled]
    adaline_online = Adaline()

    print(len(train_X_std))

    for i in range(0, len(train_X_std), 5):
        # sample 5 datapoints from train_X
        samples_X = train_X_std[i : i + 5]
        samples_y = train_y[i : i + 5]

        adaline_online.online_fit(samples_X, samples_y)

        lines.set_data(
            range(1, len(adaline_online.cost_list) + 1), adaline_online.cost_list
        )
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        sleep(1)


if __name__ == "__main__":
    main()
    sleep(30000)
