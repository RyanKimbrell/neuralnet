import numpy as np
import matplotlib as plt
import random

#============================================


def main():
    """
    TESTING
    """
    x = sigmoid(5)
    print(x)


#============================================


def neural_network(input1, input2, weight1, weight2, bias):
    return sigmoid((input1 * weight1) + (input2 * weight2) + bias)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def squared_error(prediction, target):
    return (prediction - target) ** 2


#============================================

if __name__ == "__main__":
    main()