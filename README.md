SVM
===

This is a simple implementation of a support vector machine in Python, using the packages NumPy, Matplotlib, and CVXOPT (for the convex optimization). For this project, I studied linear SVMs using Lutz Hamel's "Knowledge Discovery with Support Vector Machines" (2009, Univ. of Rhode Island), which is an excellent introduction to SVMs and perhaps the most accessible treatise published since SVMs were first introduced in 1995 (Cortes, Corinna; and Vapnik, Vladimir N.; "Support-Vector Networks", Machine Learning, 20, 1995). I also found Mathieu Blondel's SVM implementation (https://gist.github.com/mblondel/586753) helpful in figuring out how exactly to implement the weights and bias calculations involved in the convex optimization. My own motivation for implementing a SVM was to gain a deeper understanding of how this kind of machine learning classifier works (both mathematically and programmatically), so that in working with pre-built SVMs, such as that of Scikit-Learn, I'd better understand how to fine-tune SVMs for more precise data analysis.

Currently, this SVM only supports linearly separable datasets, and thus only has the associated linear kernel function. I plan to incorporate polynomial, Gaussian, and RBF kernel functions, and provide support for datasets with 3+ features (i.e. higher-dimensional sets). There are other factors, such as slack variables for soft margins, which would also be worth implementing. For the convex optimization, I may eventually replace CVXOPT's quadratic solver with the standard technique used in SVMs nowadays, sequential minimal optimization.

For now, though, this provides a simple demonstration of:

1) Finding maximum-margin decision boundaries and supporting hyperplanes through convex optimization.

2) Fitting a SVM to a set of linearly separable training datapoints, and predicting the labels of new ones (i.e. a test set in this implementation).

3) Dynamically plotting the results using Matplotlib's animation module.
