""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression



def train_model():
    data = load_digits()
    num_trials = 10
    train_percentages = range(5, 90, 5)
    test_accuracies = numpy.zeros(len(train_percentages))

    # train models with training percentages between 5 and 90 (see
    # train_percentages) and evaluate the resultant accuracy for each.
    # You should repeat each training percentage num_trials times to smooth out
    # variability.
    # For consistency with the previous example use
    # model = LogisticRegression(C=10**-10) for your learner

    # TODO: your code here
    while num_trials<0:
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,train_size=0.5)
        model = LogisticRegression(C=10**-10)
        model.fit(X_train, y_train)

        print("Train accuracy %f" %model.score(X_train, y_train))
        print("Test accuracy %f"%model.score(X_test, y_test))
    num_trials-=1
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    display_digits()
    # train_model()
