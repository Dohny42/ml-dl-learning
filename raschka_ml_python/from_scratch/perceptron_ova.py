"""
Let's use the perceptron model inside the world of multiple class classification.

The strategy is called One-vs-All: We'll train N perceptrons acting as binary
classifier between one class and all the other ones. N is number of classes.

We gotta create seperate datasets for each corresponding classifier.
Then we train them and in the prediction stage we just select the maximum of
predicted outputs.

Conclusion: OvA model with simple perceptron performs with perfect acc on the
setosa vs all dataset but poorly on the other ones. I tried to incorporate
train vs test, random init, shuffling the data, changing the "chunk" order...
but all failed to deliver perfect acc.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron


def main():
    # crate sub-datasets for each classifier
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    """ train_test_ratio = 0.8

    setosa = df.iloc[:50, :]
    train_setosa = setosa.sample(frac=train_test_ratio)
    test_setosa = setosa.drop(train_setosa.index)

    versicolor = df.iloc[50:100, :]
    train_versicolor = versicolor.sample(frac=train_test_ratio)
    test_versicolor = versicolor.drop(train_versicolor.index)

    virginica = df.iloc[100:, :]
    train_virginica = virginica.sample(frac=train_test_ratio)
    test_virginica = virginica.drop(train_virginica.index)

    train_df = pd.concat([train_versicolor, train_setosa, train_virginica])
    test_df = pd.concat([test_versicolor, test_setosa, test_virginica])

    train_df_setosa = pd.concat([train_setosa, train_versicolor, train_virginica])
    train_df_versicolor = pd.concat([train_versicolor, train_setosa, train_virginica])
    train_df_virginica = pd.concat([train_virginica, train_versicolor, train_setosa])

    test_df_setosa = pd.concat([test_setosa, test_versicolor, test_virginica])
    test_df_versicolor = pd.concat([test_versicolor, test_setosa, test_virginica])
    test_df_virginica = pd.concat([test_virginica, test_versicolor, test_setosa])

    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    train_X_setosa = train_df_setosa.iloc[:, 0:4].values
    train_X_versicolor = train_df_versicolor.iloc[:, 0:4].values
    train_X_virginica = train_df_virginica.iloc[:, 0:4].values

    train_y_setosa = train_df_setosa.iloc[:, 4].values
    train_y_setosa = np.where(train_y_setosa == "Iris-setosa", -1, -1)

    train_y_versicolor = train_df_versicolor.iloc[:, 4].values
    train_y_versicolor = np.where(train_y_versicolor == "Iris-versicolor", -1, -1)

    train_y_virginica = train_df_virginica.iloc[:, 4].values
    train_y_virginica = np.where(train_y_virginica == "Iris-virginica", -1, 1)

    # -------------------------------------------------------------------------

    test_X_setosa = test_df_setosa.iloc[:, 0:4].values
    test_X_versicolor = test_df_versicolor.iloc[:, 0:4].values
    test_X_virginica = test_df_virginica.iloc[:, 0:4].values

    test_y_setosa = test_df_setosa.iloc[:, 4].values
    test_y_setosa = np.where(test_y_setosa == "Iris-setosa", 1, -1)

    test_y_versicolor = test_df_versicolor.iloc[:, 4].values
    test_y_versicolor = np.where(test_y_versicolor == "Iris-versicolor", 1, -1)

    test_y_virginica = test_df_virginica.iloc[:, 4].values
    test_y_virginica = np.where(test_y_virginica == "Iris-virginica", 1, -1)

    train_X = train_df.iloc[:, 0:4].values

    train_y = train_df.iloc[:, 4].values
    train_y_setosa = np.where(train_y == "Iris-setosa", 1, -1)
    train_y_versicolor = np.where(train_y == "Iris-versicolor", 1, -1)
    train_y_virginica = np.where(train_y == "Iris-virginica", 1, -1)

    test_X = test_df.iloc[:, 0:4].values

    test_y = test_df.iloc[:, 4].values
    test_y_setosa = np.where(test_y == "Iris-setosa", 1, -1)
    test_y_versicolor = np.where(test_y == "Iris-versicolor", 1, -1)
    test_y_virginica = np.where(test_y == "Iris-virginica", 1, -1) """

    train_X = df.iloc[:, 0:4].values

    train_y = df.iloc[:, 4].values
    train_y_setosa = np.where(train_y == "Iris-setosa", 1, -1)
    train_y_versicolor = np.where(train_y == "Iris-versicolor", 1, -1)
    train_y_virginica = np.where(train_y == "Iris-virginica", 1, -1)

    perceptron_setosa = Perceptron().fit(train_X, train_y_setosa)
    perceptron_versicolor = Perceptron().fit(train_X, train_y_versicolor)
    perceptron_virginica = Perceptron().fit(train_X, train_y_virginica)

    plt.plot(
        range(1, len(perceptron_setosa.errors_list) + 1),
        perceptron_setosa.errors_list,
        marker="o",
    )
    plt.plot(
        range(1, len(perceptron_versicolor.errors_list) + 1),
        perceptron_versicolor.errors_list,
        color="red",
        marker="^",
    )
    plt.plot(
        range(1, len(perceptron_virginica.errors_list) + 1),
        perceptron_virginica.errors_list,
        color="green",
        marker="x",
    )
    plt.xlabel("epochs")
    plt.ylabel("errors")
    plt.show()

    """ setosa_test_acc = perceptron_setosa.test(test_X_setosa, test_y_setosa)
    versicolor_test_acc = perceptron_versicolor.test(
        test_X_versicolor, test_y_versicolor
    )
    virginica_test_acc = perceptron_virginica.test(test_X_virginica, test_y_virginica)

    print(f"{setosa_test_acc=}\n{versicolor_test_acc=}\n{virginica_test_acc=}") """


if __name__ == "__main__":
    main()
