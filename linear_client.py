#!/usr/bin/env python

###########################################################################
# Name: Evan Chow
# Date: 1/3/2014
# Contact: echow4@outlook.com
# Course project? no
# Description: Demo client for linear SVM in svm.py.
###########################################################################


import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cvxopt import solvers, matrix
from svm import svm # SVM programmed in svm.py


###########################################################################
# main() function for standalone (non-import) use.
# The following code generates two clusters (w/two labels, blue for +1 and
# red for -1) of training data using NumPy's random.multivariate_normal()
# function, and fits a SVM classifier to that dataset. Subsequently,
# a test dataset is similarly generated, and the SVM is used to predict
# the labels for the new data. All of this is plotted in real-time using
# Matplotlib's set of animation modules.
###########################################################################


def main():

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

    #######################################################################
    # clust_input()
    # Custom input for the training data cluster centers, i.e. where
    # the avg of each class' data points will be.
    #######################################################################

    def clust_input():
        default = [[3, 3],[7, 7]]
        while True:
            clust_custom = raw_input("Enter custom cluster centers? [Y/n] ")
            if clust_custom == "Y":
                c1_x = raw_input("X-coordinate of class 1's center: ")
                c1_y = raw_input("Y-coordinate of class 1's center: ")
                c2_x = raw_input("X-coordinate of class 2's center: ")
                c2_y = raw_input("Y-coordinate of class 2's center: ")
                try:
                    return [[float(x) for x in i] 
                    for i in ([c1_x, c1_y],[c2_x, c2_y])]
                except ValueError:
                    if not c1_x or not c1_y or not c2_x or not c2_y:
                        confirm = raw_input("You forgot to enter a value. "
                                            "Continue with defaults? [Y/n] ")
                        if confirm == "Y":
                            return default
                        else:
                            print "Please try input again."
                    else:
                        print ("Sorry, one of the values was not a number. "
                            "Please try input again.")
            else:
                    return default

    #######################################################################
    # cov_input()
    # Custom input for the covariance matrix, which (basically) affects
    # how spread out the training data will be.
    #######################################################################

    def cov_input():
        default = [[0.6, 0], [0, 0.6]]
        while True:
            cov_custom = raw_input("Enter a custom covariance matrix? [Y/n] ")
            if cov_custom == "Y":
                print "Enter your matrix values 1-4 in format:"
                print "[[1, 2]\n [3, 4]]"
                m1 = raw_input("1: ")
                m2 = raw_input("2: ")
                m3 = raw_input("3: ")
                m4 = raw_input("4: ")
                try:
                    return [[float(x) for x in i] for i in ([m1, m2],[m3, m4])]
                except ValueError:
                    if not m1 or not m2 or not m3 or not m4:
                        confirm = raw_input("You forgot to enter a value. "
                            "Continue with defaults? [Y/n] ")
                        if confirm == "Y":
                            return default
                        else:
                            print "Please try input again."
                    else:
                        print ("Sorry, one of the values was not a number. "
                                "Please try input again.")
            else:
                return default

    #######################################################################
    # nodes_input()
    # Custom input for number of nodes, i.e. data points for an individual
    # class. Because there are two classes for both the training and
    # testing data, you'll end up with n * 4 datapoints.
    #######################################################################

    def nodes_input():
        default = 100
        while True:
            node_custom = raw_input("Enter custom # of nodes? [Y/n] ")
            if node_custom == "Y":
                n = raw_input("Number of nodes: ")
                try:
                    return int(n)
                except ValueError:
                    print "Input was not a number; please try again."
            else:
                return default

    """ Example of data analysis/visualization with SVM. """
    
    print "If you just want a quick SVM demo, leave the following blank."
    cluster_ctrs = clust_input()
    cov_matrix = cov_input()
    n = nodes_input()

    # Generate and format datapoints.
    cluster_c1, cluster_c2 = [i for i in cluster_ctrs]
    print "Cluster centers:", cluster_c1, cluster_c2
    print "Covariance matrix", cov_matrix
    print "Number of nodes", n
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
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    fig_limit = np.sum(cluster_ctrs) / 2.0 # plot size scales to data
    ax.set_ylim(0, fig_limit)
    ax.set_xlim(0, fig_limit)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Linear SVM Demo')
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
    print "Shape:\n%s\n" % (y_train.shape)
    print "--TESTING DATA--"
    print "x_test is:"
    print "Type:", type(x_test)
    print "Shape:\n%s\n" % (x_test.shape)


main()