## Getting Started
In this repository I've implemented the QDA (Quadratic Discriminant Analysis) and LDA (Linear Discriminant Analysis) classifiers which were trained using Fisher's Iris Data Set.

To run the classifiers, just run the `qda.py` or `lda.py` files
```bash
$ python qda.py
$ python lda.py
```

## About the Data Set
With this data set, I was able to classify a new iris instance to one of these three types:
1. Iris Setosa
2. Iris Versicolour
3. Iris Virginica

To classify a new instance, there were four main features that were taken into consideration:
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm

The data set used for training and testing can be found in `data/iris.data`. All other .data files are simply variations of the original data which were generated to determine the most important features during feature extraction.

## About the Classification Models
#### QDA (Quadratic Discriminant Analysis)
All classification models are trying to determine the classification boundary between classes. To do this, we first need to train our model to better understand the information about each class (Setosa, Versicolour, Virginica)

We first calculate the mean and the covariance of each class. When doing this, we treat each iris instance as a 4-dimensional vector to take into account for the four features we are using for classification.

Then, to classify this new iris instance we apply the Gaussian Naive Bayes instance to determine which class is best fit (probabilistically) for this new instance.

Note: this means that to draw the classification boundary between classes, we simply need to identify the points at which any new iris-instance would be equally probably to belong to all three classes.

#### LDA (Linear Discriminant Analysis)
The LDA classifier is simply a special case of the QDA classifier. In the LDA classifier, the covariance matrices of all classes are equal.

To achieve this, I simply calculated for the average covariance matrix among all classes and used that matrix when plugging new instances back into the multivariate gaussian probability density function.

Note: in LDA, all classification boundaries are linear.

