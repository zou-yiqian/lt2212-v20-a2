# LT2212 V20 Assignment 2

Part 1

In the part 1, I made the dataset lowercase, remove all the punctuations and removed all the numbers.


Part 2

I used SVD in this part.


Part 3

K-nearest Neighbours and SVC were chosen as two classififies.

Part 4

1)Beause of running time, it is still running.

2)The result is shown below.
When n_dim = 5, Classifier = K-nearest, weighted avg = 0.11 0.10 0.10
When n_dim = 5, Classifier = SVC, weighted avg = 0.16 0.14 0.11
When n_dim = 10, Classifier = K-nearest, weighted avg = 0.17 0.15 0.15
When n_dim = 10, Classifier = SVC, weighted avg = 0.21 0.21 0.19
When n_dim = 25, Classifier = K-nearest, weighted avg = 0.22 0.20 0.20
When n_dim = 25, Classifier = SVC, weighted avg = 0.34 0.34 0.33
When n_dim = 50, Classifier = K-nearest, weighted avg = 0.28 0.26 0.26
When n_dim = 50, Classifier = SVC, weighted avg = 0.51 0.50 0.50

As the value of n_dim increases, the weighted avg increases. The result of SVC is more accurate than K-nearest.

Part Bonus

1)Beause of running time, it is still running.

2)The result is shown below.
When n_dim = 5, Classifier = K-nearest, weighted avg = 0.11 0.10 0.10
When n_dim = 5, Classifier = SVC, weighted avg = 0.18 0.15 0.13
When n_dim = 10, Classifier = K-nearest, weighted avg = 0.16 0.15 0.14
When n_dim = 10, Classifier = SVC, weighted avg = 0.21 0.21 0.19
When n_dim = 25, Classifier = K-nearest, weighted avg = 0.23 0.20 0.21
When n_dim = 25, Classifier = SVC, weighted avg = 0.34 0.34 0.33
When n_dim = 50, Classifier = K-nearest, weighted avg = 0.28 0.25 0.25
When n_dim = 50, Classifier = SVC, weighted avg = 0.51 0.50 0.50

The result of PCA is similar to SVD. As the value of n_dim increases, the weighted avg increases. The result of SVC is more accurate than K-nearest.
