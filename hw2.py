'''
Problem 1. Consider the same logistic regression example that was solved in "logistic-regression.pdf" (under module 4) using the built-in method fmin_tnc.

(a) Code up your own gradient descent optimizer with backtracking line search. 

(b) Show that your code from part (a) can get very close to the same solution that was found in "logistic-regression.pdf".

(c) Collect the sequence of weight vectors that your descent method uses in each of its steps, and plot them, along with the contours of the loss function,
as we did in module 4, slides 8 and 14. (You do not have to use the same fonts/colours etc. as long as you show the contours, and the steps). 

(d) for the same problem, code up newton's method using the exact Hessian derived in lecture. Show the steps it takes, 
as you did in part (c) for gradient descent. Compare and contrast -- does it converge to the right solution? 
does it take more steps? use less overall time? does adding backtracking line search help or hurt the convergence?

'''

from cmath import isnan
import math
from turtle import shape
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.optimize import fmin_tnc

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prob(theta, x):
    return sigmoid(np.dot(x, theta))

def objective(theta, x, y): # Equation in slide 39
    # We calculate the objective function value for all the training set.
    p = prob(theta, x)
    a = np.log(p)
    b = np.log(1-p)
    return - np.sum( y * a + (1 - y) * b)

def gradient(theta, x, y): # From equation (4.8) of slides. xT (mu(w) - y)
    return np.dot(x.T, sigmoid(np.dot(x, theta)) - y)

def gradient_one_point(theta, x, y): # From equation (4.8) of slides. xT (mu(w) - y)
    a = x.T
    b =  sigmoid(np.dot(x, theta))
    c = y.reshape(-1,1)
    return np.dot(a,b - c)

def fit(x, y, theta):
    return fmin_tnc(func=objective, x0=theta, fprime=gradient, args=(x, y))[0]


def fit_backtrack(x, y, theta, alpha, beta):

    t_star = theta
    t = 1
    i = 1

    #print(f"Gradient shape: {gradient(t_star, x, y).shape}")
    #print(f"np.dot(x, theta) shape: {(np.dot(x, theta)).shape}")
    #print(f"y shape: {y.shape}")
    #print(f"(sigmoid(np.dot(x, theta)) shape: {(sigmoid(np.dot(x, theta))).shape}")
    #print(f"sigmoid(np.dot(x, theta)) - y shape: {(sigmoid(np.dot(x, theta)) - y.reshape(-1,1)).shape}")
    #print(f"x.T shape: {(x.T.shape)}")
    #print(f"grad shape: {(np.dot(x.T, (sigmoid(np.dot(x, theta)) - y.reshape(-1,1)))).shape}")
    #print(t_star.shape)
    #print((t_star + t * gradient(t_star, x, y)).shape)
    #print(f"t * gradient_one_point(t_star, x, y): {t * gradient_one_point(t_star, x, y)}")
    #print(f"t_star: {t_star}")
    #print(f"t_star + t * gradient_one_point(t_star, x, y): {t_star + t * gradient_one_point(t_star, x, y)}") 

    a = objective(t_star + t * gradient_one_point(t_star, x, y) / np.linalg.norm(gradient_one_point(t_star, x, y)), x, y)
    b = objective(t_star , x, y) + alpha * t * np.linalg.norm(gradient_one_point(t_star , x, y))**2
    while  a - b > 0 and i < 2000:
        t *= beta
        a = objective(t_star + t * gradient_one_point(t_star, x, y) / np.linalg.norm(gradient_one_point(t_star, x, y)), x, y)
        b = objective(t_star , x, y) - alpha * t * np.linalg.norm(gradient_one_point(t_star , x, y))**2
        i += 1

        print(f"Iteration {i}: t = {t}, a = {a}, b = {b}, theta = {np.squeeze(t_star + t * gradient_one_point(t_star, x, y) / np.linalg.norm(gradient_one_point(t_star, x, y))**2).T}  Error: {a - b}")

    return np.squeeze(t_star + t * gradient_one_point(t_star, x, y) / np.linalg.norm(gradient_one_point(t_star, x, y)))


def accuracy(x, actual_classes, theta_star):
    predicted_classes = (prob(theta_star, x) >= 0.5).astype(int).flatten()
    return 100 * np.mean(predicted_classes == actual_classes)

def plot_decision_boundary(x, par):
    x_values = [np.min(x[:, 1] - 5), np.max(x[:, 2] + 5)]
    y_values = - (par[0] + np.dot(par[1], x_values)) / par[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    plt.xlabel('First Exam')
    plt.ylabel('Second Exam')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("marks.txt")

    X = data.iloc[:, :-1]
    X = np.c_[np.ones((X.shape[0], 1)), X] ## augment with column of ones
    print(f"X Shape: {X.shape}") # (99,3)
    
    # y = target values, last column of the data frame
    y = data.iloc[:, -1].to_numpy()
    print(f"y Shape: {y.shape}") # (99,)

    admitted = data.loc[y == 1]
    not_admitted = data.loc[y == 0]

    #plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    #plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    #plt.legend()
    #plt.show()

    #theta_star = fit(X, y, np.zeros((X.shape[1], 1)))
    #print(f"theta_star: {theta_star}")
    #print(theta_star) # (3,)
    
    #theta_star = np.array([-20, 0.1, 0.1]) # This set of thetas does not work! just for debugging
    theta_star_backtrack = fit_backtrack(X, y, np.zeros((X.shape[1], 1)), 0.4, 0.99)
    print(f"theta_star_backtrack: {theta_star_backtrack}")


    print(accuracy(X, y, theta_star_backtrack))
    
    plot_decision_boundary(X, theta_star_backtrack)