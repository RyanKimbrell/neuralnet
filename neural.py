from matplotlib import pyplot as plt
import numpy as np


#============================================


def main():
    """
    TESTING
    """

    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()

    data, mystery_data = get_data()

    scatter_plot(data)

#============================================


#===================
# NEURAL NET FUNCTIONS
#===================
def get_data():
    # each point is length, width, type
    data =[[3.0,  1.5,  1],
           [2.0,  1.0,  0],
           [4.0,  1.5,  1],
           [3.0,  1.0,  0],
           [3.5,  0.5,  1],
           [2.0,  0.5,  0],
           [5.5,  1.0,  1],
           [1.0,  1.0,  0]]

    mystery_data = [4.5, 1.0]

    return data, mystery_data

def neural_network(input1, input2, weight1, weight2, bias):
    return sigmoid((input1 * weight1) + (input2 * weight2) + bias)


def training_loop(data, w1, w2, b):
    for i in range(100):
        ri = np.random.randint(len(data))
        point = data[ri]
        print(point)

        z = (point[0] * w1) + (point[1] * w2) + b
    
        prediction = sigmoid(z)

        target = point[2]
        cost = squared_error(prediction, target)

        d_cost_pred = d_squared_error(prediction, target)
        dpred_dz = sigmoid_prime(z)

        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1

        dcost_dz = d_cost_pred * dpred_dz

        dcost_dw1 =  dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2 


#===================
# MATH FUNCTIONS
#===================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    # the derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


def squared_error(prediction, target):
    return np.square(prediction - target)


def d_squared_error(prediction, target):
    return 2 * (prediction - target)


#===================
# GRAPHING FUNCTIONS
#===================
def scatter_plot(data):

    #visual formatting
    plt.axis([0, 6, 0, 6])
    plt.grid()

    # plot the data, color appropriately
    for i in range(len(data)):
        point = data[i]
        color = "r"
        if point[2] == 0:
            color = "b"
        plt.scatter(point[0], point[1], c=color)

    # show the graph
    plt.show()
    

#============================================

if __name__ == "__main__":
    main()