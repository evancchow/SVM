""" An implementation and demo of a 2-D linear Support Vector Machine.
    Contact information: echow4@outlook.com """

import sys, random, time
sys.path.append('C:\Python27\Lib\site-packages')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cvxopt import solvers, matrix

class svm():
    """ A Support Vector Machine classifier object. """

    # Linear kernel
    def linear_k(x1, x2):
        return np.dot(x1, x2)
    
    # Initiate classifier object
    def __init__(self, kernel=linear_k):
        self.kernel=kernel
        
    # Fits training x's to training y's.
    # x is a numpy array of shape (# of training samples, # of dimensions*)
    # y is a numpy array of shape (# of training samples,)
    # *For example, in 2-D space, needs 2 columns for each point (x1,x2)
    def fit(self, x, y):
        # Store training x, y in the classifier; get x's dimensions
        self.x_train = x
        self.y_train = y
        y_dim, x_dim = x.shape

        # Kernel matrix; out product of y's
        F = np.zeros((y_dim, y_dim))
        for i in xrange(y_dim):
            for j in xrange(y_dim):
                F[i][j] = self.kernel(x[i], x[j])
        ymatrix = np.outer(y, y)
        
        # P, q, G, h, A, b matrices
        P = matrix(F * ymatrix)
        q = matrix(np.ones(y_dim) * -1)
        G = matrix(np.diag(np.diag(np.ones((y_dim, y_dim))) * -1))
        h = matrix(np.zeros(y_dim).reshape(y_dim, -1))
        A = matrix(y.reshape(1,y_dim).astype(float))
        b = matrix(0.0)
        
        # Get Lagrangian multipliers, support vectors
        alphas = solvers.qp(P, q, G, h, A, b)['x']
        svects = np.array([(idx, a) 
            for idx, a in enumerate(alphas) if a > 1e-5])
        self.a_ind = svects[:, 0]
        self.a = svects[:, 1]
        self.x = np.array([i for idx, i in enumerate(x) if idx in self.a_ind])
        self.y = y[self.a_ind.astype(int)]
        
        # Wights and bias calculations based on Mathieu Blondel's SVM code.
        # https://gist.github.com/mblondel/586753
        # Compute optimal w, the weights
        self.w = np.zeros((1, 2))
        for i in xrange(len(self.a)):
            self.w += self.a[i] * self.y[i] * x[self.a_ind[i]]
        
        # Compute optimal b, the bias
        self.bias = 0.0
        tf = [True if a > 1e-5 else False for a in alphas]
        for i in xrange(len(self.a_ind)):
            self.bias += self.y[i]
            tf_values = [x for ix, x in enumerate(F[self.a_ind[i]]) if tf[ix] == True]
            self.bias -= np.sum(self.a * self.y * tf_values)
        self.bias /= len(self.a)
        return
        
    # Generates predictions for a testing set x.
    # x is a numpy array of size (# of points to test, # of dimensions)
    # See above for notes about # of dimensions.
    def predict(self, x):
        def classify(wopt, xi, bopt):
            return np.sign(np.dot(wopt, xi) - bopt)
        b = -1 * self.bias
        w = self.w
        return [1 if classify(w, i, b) >= 0 else -1 for i in x]

    # Plots the maximum-margin decision boundary between 0 and 1 on the x-axis.
    # Also plots supporting hyperplanes, which go through the support vectors.
    def plot_boundary(self):
        def plot(x, w, b, c=0):
            return (c-b-w[0][0]*x) / w[0][1]
        b = self.bias
        w = self.w
        x = [0, 10]
        c = [0, 1, -1]

        # Plot maximum-margin decision boundary and supporting hyperplanes
        plt.plot([x[0], x[1]], [plot(x[0], w, b), plot(x[1], w, b)])
        plt.plot([x[0], x[1]], [plot(x[0], w, b, c[1]), 
            plot(x[1], w, b, c[1])], 'k--')
        plt.plot([x[0], x[1]], [plot(x[0], w, b, c[2]), 
            plot(x[1], w, b, c[2])], 'k--')
    
    # Plots testpoints, colorcoding Blue/Red for classes 1/2 given predictions.
    # Built-in function. Not used for the below main() but useful for OOP.
    # I.e. after plot_boundary(), clf.plot_predictions(x_test, predictions) 
    def plot_predictions(self, testpoints, predictions):
        for idx, i in enumerate(testpoints):
            if predictions[idx] >= 0: 
                plt.scatter(i[0], i[1], color='b',marker='x')
            else:
                plt.scatter(i[0], i[1], color='r', marker='x')

if __name__=="__main__":
    """ Generate n training data points (classes c1, c2) w/specific centers. 
        The most important parameters are cluster_"""

    # Cluster centers, covariance matrix, and # of training data points.
    cluster_centers = [[3, 3], [7,  7]]
    cov_matrix = [[0.6, 0], [0, 0.6]]
    n = 100

    # Generate data points, and do some preprocessing.
    cluster_c1 = random.choice(cluster_centers)
    cluster_c2 = random.choice([i for i in cluster_centers if i != cluster_c1])
    print "Cluster centers:", cluster_c1, cluster_c2
    x_train_c1 = np.random.multivariate_normal(cluster_c1, cov_matrix, n)
    y_train_c1 = np.ones(n)
    x_train_c2 = np.random.multivariate_normal(cluster_c2, cov_matrix, n)
    y_train_c2 = np.ones(n) * -1
    x_train = np.vstack((x_train_c1, x_train_c2))
    y_train = np.hstack((y_train_c1, y_train_c2))

    # local function to random order of rows in x_train, y_train.
    def randomize(x_train, y_train):
        aggregated = np.concatenate((x_train, y_train.T.reshape(-1, 1)), axis=1)
        np.random.shuffle(aggregated)
        return aggregated[:, 0:2], aggregated[:, 2]

    # Finish preprocessing code.
    x_train, y_train = randomize(x_train, y_train)
    y_train_idx = np.array([(idx, predict)
        for idx, predict in enumerate(y_train)])

    """ Example of data analysis/visualization with above SVM. """
    # Read training points. Classes +1 = blue, -1 = red
    def gen_train():
        for a, b, idx, predict in zip(x_train[:, 0], x_train[:, 1], 
            y_train_idx[:, 0],  y_train_idx[:, 1]):
            yield a, b, idx, predict

    # Plot training points based on class.
    def plot_train(gen_train):
        a, b, idx, predict = (i for i in gen_train)
        if predict == 1:
            x_data_c1.append(a)
            y_data_c1.append(b)
            plot_c1.set_data(x_data_c1, y_data_c1)
            return plot_c1
        else:
            x_data_c2.append(a)
            y_data_c2.append(b)
            plot_c2.set_data(x_data_c2, y_data_c2)
            return plot_c2

    # Read test points. Classes +1 = blue, -1 = red
    def gen_testpoints():
        for c, d, idx, prediction in zip(x_test[:, 0], x_test[:, 1],  
            predictions_idx[:, 0], predictions_idx[:, 1]):
            yield c, d, idx, prediction

    # Plot test points based on class.
    def plot_testpoints(gen_test):
        c, d, idx, predict = (i for i in gen_test)
        if predict == 1:
            x_test_c1.append(c)
            y_test_c1.append(d)
            plot_c1_test.set_data(x_test_c1, y_test_c1)
            return plot_c1_test
        else:
            x_test_c2.append(c)
            y_test_c2.append(d)
            plot_c2_test.set_data(x_test_c2, y_test_c2)
            return plot_c2_test

    # Empty lists for plotting
    x_data_c1, y_data_c1 = [], []
    x_data_c2, y_data_c2 = [], []
    x_test_c1, y_test_c1 = [], []
    x_test_c2, y_test_c2 = [], []

    # plot_class1(x_train_c1, x_train_c2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 10)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('SVM Data Classification')
    plot_c1, = ax.plot([], [], 'bo', ms=10)
    plot_c2, = ax.plot([], [], 'ro', ms=10)
    plot_c1_test, = ax.plot([], [], 'b+', ms=10)
    plot_c2_test, = ax.plot([], [], 'r+', ms=10)

    # Train our SVM classifier
    clf = svm()
    clf.fit(x_train, y_train)
    print "Weight vector: %s\nBias: %s" % (clf.w, clf.bias)

    # Generate n test datapoints, centered at training data cluster centers.
    n_test = 200
    mean_test = 0.5 * np.add(cluster_c1, cluster_c2)
    cov_test = [[1, 0], [0, 1]]
    a, b = np.random.multivariate_normal(mean_test, cov_test, n_test).T
    x_test = np.array([(a[i], b[i]) for i in xrange(n_test)])
    predictions = clf.predict(x_test)
    predictions_idx = np.array([(idx, predict)
        for idx, predict in enumerate(clf.predict(x_test))])

    # Visualize results, and write to stdout
    anim_train = animation.FuncAnimation(fig, plot_train, 
        gen_train, blit=False,interval=1, repeat=False)
    anim_test = animation.FuncAnimation(fig, plot_testpoints, 
        gen_testpoints, blit=False,interval=1, repeat=False)
    clf.plot_boundary()
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

    plt.show()