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
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append('C:\Python27\Lib\site-packages')
from cvxopt import solvers, matrix


class svm():
    """ A Support Vector Machine classifier object. """

    # Linear kernel function.
    def linear_k(x1, x2):
        return np.dot(x1, x2)

    # Initiate classifier object.
    def __init__(self, kernel=linear_k):
        self.kernel=kernel
    
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

###########################################################################
# main() function for standalone (non-import) use.
# The following code generates two clusters (w/two labels, blue for +1 and
# red for -1) of training data using NumPy's random.multivariate_normal()
# function, and fits a SVM classifier to that dataset. Subsequently,
# a test dataset is similarly generated, and the SVM is used to predict
# the labels for the new data. All of this is plotted in real-time using
# Matplotlib's set of animation modules.
###########################################################################

if __name__ == "__main__":

    #######################################################################
    # randomize(x_train, y_train)
    # Randomize the order of rows in the training data. This to allow for 
    # more integrated real-time plotting; given how the training data is
    # generated, all datapoints of one class will be plotted first (and
    # then all points of the other class), unless this function is applied
    # to allow both clusters to be plotted near-simultaneously.
    #######################################################################

    def randomize(x_train, y_train):
        merged = np.concatenate((x_train, y_train.T.reshape(-1, 1)), axis=1)
        np.random.shuffle(merged)
        return merged[:, 0:2], merged[:, 2]

    #######################################################################
    # gen_train()
    # Read in the training points, with the label +1 (blue) or -1 (red).
    #######################################################################

    def gen_train():
        for a, b, idx, predict in zip(x_train[:, 0], x_train[:, 1], 
            y_train_idx[:, 0],  y_train_idx[:, 1]):
            yield a, b, idx, predict

    #######################################################################
    # plot_train(gen_train)
    # Plot training points (read in with gen_train) based on class.
    #######################################################################

    def plot_train(gen_train):
        a, b, idx, predict = (i for i in gen_train)
        if predict == 1:
            xplt_train_c1.append(a)
            yplt_train_c1.append(b)
            plot_c1_train.set_data(xplt_train_c1, yplt_train_c1)
            return plot_c1_train
        else:
            xplt_train_c2.append(a)
            yplt_train_c2.append(b)
            plot_c2_train.set_data(xplt_train_c2, yplt_train_c2)
            return plot_c2_train

    #######################################################################
    # gen_test()
    # Read in the generated test points, with the labels +1/-1 as above.
    #######################################################################

    def gen_test():
        for c, d, idx, predict in zip(x_test[:, 0], x_test[:, 1],  
            predictions_idx[:, 0], predictions_idx[:, 1]):
            yield c, d, idx, predict

    #######################################################################
    # plot_test(gen_test)
    # Plot the test points based on class.
    #######################################################################
    
    def plot_test(gen_test):
        c, d, idx, predict = (i for i in gen_test)
        if predict == 1:
            xplt_test_c1.append(c)
            yplt_test_c1.append(d)
            plot_c1_test.set_data(xplt_test_c1, yplt_test_c1)
            return plot_c1_test
        else:
            xplt_test_c2.append(c)
            yplt_test_c2.append(d)
            plot_c2_test.set_data(xplt_test_c2, yplt_test_c2)
            return plot_c2_test

    """ Example of data analysis/visualization with SVM. """

    # Cluster centers, covariance matrix, and # of training data points.
    cluster_centers = [[3, 3], [7,  7]]
    cov_matrix = [[0.6, 0], [0, 0.6]]
    n = 100

    # Generate and format datapoints.
    cluster_c1 = random.choice(cluster_centers)
    cluster_c2 = random.choice([i for i in cluster_centers if i != cluster_c1])
    print "Cluster centers:", cluster_c1, cluster_c2
    x_train_c1 = np.random.multivariate_normal(cluster_c1, cov_matrix, n)
    y_train_c1 = np.ones(n)
    x_train_c2 = np.random.multivariate_normal(cluster_c2, cov_matrix, n)
    y_train_c2 = np.ones(n) * -1
    x_train = np.vstack((x_train_c1, x_train_c2))
    y_train = np.hstack((y_train_c1, y_train_c2))

    # Finish preprocessing code.
    x_train, y_train = randomize(x_train, y_train)
    y_train_idx = np.array([(idx, predict)
        for idx, predict in enumerate(y_train)])

    # Empty lists to hold plot train/test data.
    xplt_train_c1, yplt_train_c1 = [], []
    xplt_train_c2, yplt_train_c2 = [], []
    xplt_test_c1, yplt_test_c1 = [], []
    xplt_test_c2, yplt_test_c2 = [], []

    # Set up figure.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig_limit = np.sum(cluster_centers) / 2.0 # plot size scales to data
    ax.set_ylim(0, fig_limit)
    ax.set_xlim(0, fig_limit)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('SVM Classification Demo')
    plot_c1_train, = ax.plot([], [], 'bo', ms=10)
    plot_c2_train, = ax.plot([], [], 'ro', ms=10)
    plot_c1_test, = ax.plot([], [], 'b+', ms=10)
    plot_c2_test, = ax.plot([], [], 'r+', ms=10)
    plot_c1_train.set_label('Class 1, Train')
    plot_c2_train.set_label('Class 2, Train')
    plot_c1_test.set_label('Class 1, Test')
    plot_c2_test.set_label('Class 2, Test')
    plt.legend(loc=2, fontsize='small')

    # Build and train our SVM classifier.
    clf = svm()
    clf.fit(x_train, y_train)
    print "Weight vector: %s\nBias: %s" % (clf.w, clf.bias)

    # Generate test points & predictions.
    # n_test: # test points == # of training points
    # mean_test: test points centered at average of train. cluster centers
    # cov_test: minimal covariance, since need linearly separable data.
    n_test = n * 2
    mean_test = 0.5 * np.add(cluster_c1, cluster_c2)
    cov_test = [[1, 0], [0, 1]]
    a, b = np.random.multivariate_normal(mean_test, cov_test, n_test).T
    x_test = np.array([(a[i], b[i]) for i in xrange(n_test)])
    predictions = clf.predict(x_test)
    predictions_idx = np.array([(idx, predict)
        for idx, predict in enumerate(clf.predict(x_test))])

    # Visualize results, and write to stdout.
    # animation.FuncAnimation's interval attribute is the # of milliseconds
    # between animation events, i.e. plotting points.
    anim_train = animation.FuncAnimation(fig, plot_train, 
        gen_train, blit=False,interval=1, repeat=False)
    anim_test = animation.FuncAnimation(fig, plot_test, 
        gen_test, blit=False,interval=1, repeat=False)
    clf.plot_boundary()
    plt.show()
    print "\n--TRAINING DATA--"
    print "x_train is:"
    print "Type:", type(x_train)
    print "Shape:\n", x_train.shape
    print "y_train is:"
    print "Type:", type(y_train)
    print "Shape:\n", y_train.shape, '\n'
    print "--TESTING DATA--"
    print "x_test is:"
    print "Type:", type(x_test)
    print "Shape:", x_test.shape