#!/usr/bin/env python

###########################################################################
# Name: Evan Chow
# Date: 1/3/2014
# Contact: echow4@outlook.com
# Course project? no
# Description: Implementation and brief analysis/visualization of a 2-D
#   linear Support Vector Machine (SVM), as described in Ch. 7 of Hamel's
#   "Knowledge Discovery With Support Vector Machines" (2009). 
# Sources: Primarily the Hamel book. Also found Mathieu Blondel's SVM code
#   (https://gist.github.com/mblondel/586753) helpful in calculating
#   the weights and bias for the SVM's fit() function.
# Python packages used: NumPy, Matplotlib, CVXOPT
###########################################################################


import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cvxopt import solvers, matrix


class svm():
    """ A Support Vector Machine classifier object. """

    # Linear kernel function.
    def linear_k(x1, x2):
        return np.dot(x1, x2)

    # Initiate classifier object.
    def __init__(self, kernel=linear_k):
        self.kernel = kernel
    
    #######################################################################
    # fit(self, x, y)
    # Given a set of training x's (points) and y's (predictions, +1/-1),
    # fit() trains the SVM model using CVXOPT's quadratic program solver.
    # 1) x is a numpy array of shape (# training samples, # dimensions)
    # 2) y is a numpy array of shape (# training samples, )
    # *For example, in 2-D space, needs 2 features for each point (x1, x2).
    #######################################################################

    def fit(self, x, y):

        # Embed training data in SVM object; get x's dimensions.
        self.x_train = x
        self.y_train = y
        y_dim, x_dim = x.shape

        # Kernel matrix & outer product of y's required for the P matrix.
        F = np.zeros((y_dim, y_dim))
        for i in xrange(y_dim):
            for j in xrange(y_dim):
                F[i][j] = self.kernel(x[i], x[j])
        ymatrix = np.outer(y, y)
        
        # P, q, G, h, A, b matrices for convex optimization problem.
        P = matrix(F * ymatrix)
        q = matrix(np.ones(y_dim) * -1)
        G = matrix(np.diag(np.diag(np.ones((y_dim, y_dim))) * -1))
        h = matrix(np.zeros(y_dim).reshape(y_dim, -1))
        A = matrix(y.reshape(1,y_dim).astype(float))
        b = matrix(0.0)
        
        # Compute Lagrangian multipliers, support vectors.
        alphas = solvers.qp(P, q, G, h, A, b)['x']
        svects = np.array([(idx, a) 
            for idx, a in enumerate(alphas) if a > 1e-5])
        self.a_ind = svects[:, 0]
        self.a = svects[:, 1]
        self.x = np.array([i for idx, i in enumerate(x) if idx in self.a_ind])
        self.y = y[self.a_ind.astype(int)]
        
        # Compute optimal w, the weights.
        self.w = np.zeros((1, 2))
        for i in xrange(len(self.a)):
            self.w += self.a[i] * self.y[i] * x[self.a_ind[i]]
        
        # Compute optimal b, the bias.
        self.bias = 0.0
        tf = [True if a > 1e-5 else False for a in alphas]
        for i in xrange(len(self.a_ind)):
            self.bias += self.y[i]
            tf_values = [x for ix, x in enumerate(F[self.a_ind[i]]) 
                            if tf[ix] == True]
            self.bias -= np.sum(self.a * self.y * tf_values)
        self.bias /= len(self.a)
        return
        
    #######################################################################
    # predict(self, x)
    # Generates and returns predictions for a testing set x.
    # 1) x is a numpy array of size (# testing points, # dimensions)
    #######################################################################

    def predict(self, x):
        def classify(wopt, xi, bopt):
            return np.sign(np.dot(wopt, xi) - bopt)
        b = self.bias * -1
        w = self.w
        return [1 if classify(w, i, b) >= 0 else -1 for i in x]

    #######################################################################
    # plot_boundary(self)
    # Plots the maximum-margin decision boundary, given the weights & bias
    # are already calculated for the SVM object. Also, plot_boundary plots
    # the supporting hyperplanes, which run through the support vectors.
    # Decision boundary: black; supporting hyperplanes: dashed
    #######################################################################

    def plot_boundary(self):
        def line_eq(x, w, b, c=0):
            return (c-b-w[0][0]*x) / w[0][1]
        b = self.bias
        w = self.w
        x = [0, 10]
        c = [0, 1, -1]
        plt.plot([x[0], x[1]], [line_eq(x[0], w, b), line_eq(x[1], w, b)],
            label = "Max-Margin Hyperplane")
        plt.plot([x[0], x[1]], [line_eq(x[0], w, b, c[1]), 
            line_eq(x[1], w, b, c[1])], 'k--')
        plt.plot([x[0], x[1]], [line_eq(x[0], w, b, c[2]), 
            line_eq(x[1], w, b, c[2])], 'k--')
    
    #######################################################################
    # plot_predictions(self, testpoints, predictions)
    # Plots testing points, colorcoding blue/red for labels +1/-1 given
    # predictions. While this function (belonging to the SVM object) is 
    # not used in the demo below, it provides built-in plotting for OOP.
    #######################################################################

    def plot_predictions(self, testpoints, predictions):
        for idx, i in enumerate(testpoints):
            if predictions[idx] >= 0: 
                plt.scatter(i[0], i[1], color='b',marker='x')
            else:
                plt.scatter(i[0], i[1], color='r', marker='x')