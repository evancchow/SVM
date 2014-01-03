""" Implementation of a simple linear Support Vector Machine. """

import sys, random, time
sys.path.append('C:\Python27\Lib\site-packages')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cvxopt import solvers, matrix

class svm():
    """ A Support Vector Machine classifier object.
        Currently supports only the linear case. """

    # Linear kernel 
    def linear_k(x1, x2):
        return np.dot(x1, x2)
    
    # Gaussian kernel function. Creates lots of bugs right now.
    def gauss_k(x, y, s=0.5):
        return math.exp(-(np.linalg.norm(matrix(x)-matrix(y))**2)/(2*s**2))
    
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
        svects = np.array([(idx, a) for idx, a in enumerate(alphas) if a > 1e-5])
        self.a_ind = svects[:,0]
        self.a = svects[:,1]
        self.x = np.array([i for idx, i in enumerate(x) if idx in self.a_ind])
        self.y = y[self.a_ind.astype(int)]
        
        # Compute optimal w, the weights
        self.w = np.zeros((1,2))
        for i in xrange(len(self.a)):
            self.w += self.a[i] * self.y[i] * x[self.a_ind[i]]
        
        # Compute optimal b, the bias
        # Implementation of bias calculation based on similar part of Mathieu Blondel's SVM.
        # https://gist.github.com/mblondel/586753
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
    # Also plots the supporting hyperplanes which intersect the support vectors.
    def plot_boundary(self):
        def plot(x,w,b,c=0):
            return (c-b-w[0][0]*x) / w[0][1]
        b = self.bias
        w = self.w
        x = [0,10]
        c = [0,1,-1]

        # Plot maximum-margin decision boundary and supporting hyperplanes
        plt.plot([x[0],x[1]],[plot(x[0],w,b),plot(x[1],w,b)])
        plt.plot([x[0],x[1]],[plot(x[0],w,b,c[1]),
                              plot(x[1],w,b,c[1])],'k--')
        plt.plot([x[0],x[1]],[plot(x[0],w,b,c[2]),
                              plot(x[1],w,b,c[2])],'k--')
        plt.draw()
        time.sleep(0.1)
        plt.pause(0.0001)   
    
    # Plots testpoints, colorcoding Blue/Red for classes 1/2 given predictions. 
    def plot_predictions(self, testpoints, predictions):
        for idx, i in enumerate(testpoints):
            if predictions[idx] >= 0: 
                plt.scatter(i[0],i[1],color='b',marker='x')
                plt.draw()
                time.sleep(0.1)
                plt.pause(0.0001)  
            else:
                plt.scatter(i[0],i[1],color='r',marker='x')
                plt.draw()
                time.sleep(0.1)
                plt.pause(0.0001)  

if __name__=="__main__":

    # Generate n training data points (classes c1, c2) w/specified centers.
    means_possible = [[2,2],[7,7],[5,-4]]
    n = 30
    mean_train_c1 = random.choice(means_possible)
    mean_train_c2 = random.choice([i for i in means_possible if i != mean_train_c1])
    print "Means:", mean_train_c1, mean_train_c2
    x_train_c1 = np.random.multivariate_normal(mean_train_c1,[[1,0],[0,1]],n)
    y_train_c1 = np.ones(n)
    x_train_c2 = np.random.multivariate_normal(mean_train_c2,[[1,0],[0,1]],n)
    y_train_c2 = (np.ones(n) * -1)
    x_train = np.vstack((x_train_c1, x_train_c2))
    y_train = np.hstack((y_train_c1,y_train_c2))

    print x_train_c1
    # plot initial points
    """ Example (w/visualization) of data analysis with above SVM. """


    """ 6:06 PM: maybe have two different functions to plot both class 1
        and class 2? """
    # Plot training points. Classes +1 = blue, -1 = red
    def gen_train():
        # key to not repeating - the repeat=False in the animation object.
        for a,b in zip(x_train[:,0], x_train[:,1]):
            yield a,b
        # plt.scatter(x_train_c1[:,0], x_train_c1[:,1], color='b')
        # plt.scatter(x_train_c2[:,0], x_train_c2[:,1], color='r')

    def plot_train(gen_train):
        x, y = gen_train[0], gen_train[1]
        xdata.append(x)
        ydata.append(y)
        if len(xdata) >= plot_train.count:
            # set xdata, ydata to only be the latter half of points
            xdata = xdata[0:int(len(xdata) / 2), 0]
            xdata = xdata[0:int(len(xdata) / 2), 0]
            line2.set_data(xdata, ydata)
            return line2
        line.set_data(xdata, ydata)
        return line

    # print latter half of training set ( )
    def plot_train_2():


    xdata, ydata = [], []
    plot_train.count = len(x_train_c1)

    # plot_train(x_train_c1, x_train_c2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], 'bo', ms=10)
    line2, = ax.plot([], [], 'ro', ms=10)
    ax.set_ylim(-10,10)
    ax.set_xlim(0,20)

    ani = animation.FuncAnimation(fig, plot_train, gen_train, blit=False,\
         interval=1, repeat=False)

    # plt.xlabel('X values')
    # plt.ylabel('Y values')

    # # Train our SVM classifier
    # clf = svm()
    # clf.fit(x_train, y_train)
    # print "Weight vector: %s\nBias: %s" % (clf.w, clf.bias)

    # # Generate n test datapoints, centered at avg. of training centers.
    # n_test = 100
    # mean_test = 0.5 * np.add(mean_train_c1,mean_train_c2)
    # cov_test = [[4,0],[0,4]]
    # a, b = np.random.multivariate_normal(mean_test, cov_test, n_test).T
    # x_test = np.array([(a[i],b[i]) for i in xrange(n_test)])
    # predictions = clf.predict(x_test)

    # # Visualize results and write to stdout
    # clf.plot_boundary() ## refactor so can come before below
    # clf.plot_predictions(x_test, predictions)
    # print "--TRAINING DATA--"
    # print "x_train is:"
    # print "Type:", type(x_train)
    # print "Shape:", x_train.shape
    # print "y_train is:"
    # print "Type:", type(y_train)
    # print "Shape:", y_train.shape, '\n'
    # print "--TESTING DATA--"
    # print "x_test is:"
    # print "Type:", type(x_test)
    # print "Shape:", x_test.shape

    plt.show()