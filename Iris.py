# Import the libraries
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O
import matplotlib.pyplot as plt #plotting graphs

#Sigmoid Function
def sigmoid(z):
    """sigmoid function"""
    return 1 / (1 + np.exp(-z))

#Cost function
def cost(theta, x, y):
    """cost function"""
    h = sigmoid(x @ theta)
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h)
    )
    grad = 1 / m * ((y - h) @ x) #GRADIENT
    return cost, grad

#Fitting function
def fit(x, y, max_iter=50000, alpha=0.1):
    """logisitic classification model"""
    x = np.insert(x, 0, 1, axis=1) # x0=1
    thetas = []
    classes = np.unique(y) 
    costs = np.zeros(max_iter)

    for c in classes:
        # one vs. rest binary classification
        binary_y = np.where(y == c, 1, 0) #current class as 1 , rest as 0
        theta = np.zeros(x.shape[1])
        for epoch in range(max_iter):
            costs[epoch], grad = cost(theta, x, binary_y)
            theta += alpha * grad
        thetas.append(theta)

    return thetas, classes, costs

#Prediction function
def predict(classes, thetas, x):
    """predict class from max h(x) value"""
    x = np.insert(x, 0, 1, axis=1) #x0 = 1

    preds = [ np.argmax( [sigmoid(xi @ theta) for theta in thetas] ) for xi in x ]
    return [classes[p] for p in preds]

#Accuracy Score
def score(classes, theta, x, y):
    """calculating the accuracy score"""
    return (predict(classes, theta, x) == y).mean() *100

#Data input
iris = pd.read_csv("./input/Iris.csv") #take input
iris = iris.drop(['Id'],axis=1) #removing the id column 

iris['Species'] =iris['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}) 
#assinging integer values to the classes
data = np.array(iris)
np.random.shuffle(data) #shuffling

#splitting the data in training and testing
num_train = int(.7 * len(data))  # 70/30 train/test split
x_train, y_train = data[:num_train, :-1], data[:num_train, -1]
x_test, y_test = data[num_train:, :-1], data[num_train:, -1]

#training model with all features
thetas, classes, costs = fit(x_train, y_train)
