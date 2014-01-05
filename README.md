A Linear Support Vector Machine
===

This is an implementation of a linear support vector machine in Python, a supervised learning model used to non-probabilistically classify multidimensional datasets. For this project, I used Lutz Hamel's "Knowledge Discovery with Support Vector Machines" (2009, Univ. of Rhode Island), which is an excellent introduction to SVMs and perhaps the most accessible treatise published since SVMs were first introduced in 1995 (Cortes, Corinna; and Vapnik, Vladimir N.; "Support-Vector Networks", Machine Learning, 20, 1995). My reason for implementing a SVM was to gain a deeper understanding of how this kind of classifier works (both mathematically and programmatically), so that in working with pre-built SVMs, such as that of Scikit-Learn, I'd better understand how to fine-tune SVMs for more precise data analysis.

Currently, this SVM only supports linearly separable datasets, and thus only has a linear kernel function (f(x1, x2) = x1 dot x2). As expected, I plan to incorporate polynomial, Gaussian, and RBF kernel functions, and provide support for datasets with 3+ features (i.e. higher-dimensional sets). There are other factors, such as slack variables for soft margins, which would also be worth including. For the convex optimization, I may eventually replace CVXOPT's quadratic solver with the standard technique used in SVMs nowadays, sequential minimal optimization. This version of an SVM demonstrates:

1) Finding maximum-margin decision boundaries and supporting hyperplanes through convex optimization.

2) Fitting the SVM to a linearly separable training dataset, and predicting the labels of a testing dataset. All data is randomly generated using NumPy's random.multivariate_normal() function.

3) Dynamically plotting the results using Matplotlib's animation module. I'd like to upgrade the animation later to a higher-quality library such as NodeBox for OpenGL, Seaborn, etc.

A PNG of one possible output with 150 data points (for each class, making a total of 600 for train/test) can be found here:
https://www.dropbox.com/s/0h4xhtgtb06vvam/n150_svm.png

I also found Mathieu Blondel's SVM implementation (https://gist.github.com/mblondel/586753) in implementing the weights and bias calculations involved in the convex optimization. 