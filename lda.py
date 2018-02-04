import iris
import qda
import math
import numpy


all_probability_densities = []
preserved_sigma_matrix = None  # Assign this value right after training


############################################################
#                          Training
############################################################

# NOTE: The parameters Mu and Sigma were already estimated in QDA

# Take the average of the sigma (covariance) matrices
sigma_matrix = None
for classification in iris.CLASSIFICATIONS:
    if sigma_matrix is None:
        sigma_matrix = qda.sigma_matrices[classification]
    else:
        sigma_matrix += qda.sigma_matrices[classification]

sigma_matrix /= len(iris.CLASSIFICATIONS)
preserved_sigma_matrix = sigma_matrix


############################################################
#                          Testing
############################################################

error_rate = {}

for testing_classification in iris.CLASSIFICATIONS:
    iris_test_data = qda.cleaned_iris_test_data[testing_classification]
    iris_instances = iris_test_data.split(';')

    # Plug the instances into each classification's probability density function
    for instance in iris_instances:
        probability_densities = {}
        for classification in iris.CLASSIFICATIONS:
            mu = qda.mu_matrices[classification]
            sigma = sigma_matrix

            instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
            pdf = 1 / (math.sqrt(numpy.linalg.det(sigma)))
            exponent = ((-0.5) * (instance_matrix - mu).transpose() * numpy.linalg.inv(sigma) * (instance_matrix - mu))
            exponent = exponent.item(0)  # Turn scalar into normal real number
            pdf = pdf * math.exp(exponent)

            probability_densities[classification] = pdf

        # Choose a classification based on highest probability
        likely_classification = max(probability_densities, key=probability_densities.get)

        # Aggregate the probability densities for later analysis
        all_probability_densities.append(probability_densities)

        # Update the error rate calculation
        if testing_classification + '-total' in error_rate:
            error_rate[testing_classification + '-total'] += 1
        else:
            error_rate[testing_classification + '-total'] = 1
            error_rate[testing_classification + '-incorrect'] = 0

        if likely_classification != testing_classification:
            error_rate[testing_classification + '-incorrect'] += 1

# Calculate the Error Rate
total = 0.0
incorrect = 0.0
for classification in iris.CLASSIFICATIONS:
    total += error_rate[classification + '-total']
    incorrect += error_rate[classification + '-incorrect']

print 'LDA Testing Data Error Rate: ' + str(incorrect/total)


############################################################
#                  Testing the Training Set
############################################################

error_rate = {}

for training_classification in iris.CLASSIFICATIONS:
    iris_test_data = qda.cleaned_iris_data[training_classification]
    iris_instances = iris_test_data.split(';')

    # Plug the instances into each classification's probability density function
    for instance in iris_instances:
        probability_densities = {}
        for classification in iris.CLASSIFICATIONS:
            mu = qda.mu_matrices[classification]
            sigma = sigma_matrix

            instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
            pdf = 1 / (math.sqrt(numpy.linalg.det(sigma)))
            exponent = ((-0.5) * (instance_matrix - mu).transpose() * numpy.linalg.inv(sigma) * (instance_matrix - mu))
            exponent = exponent.item(0)  # Turn scalar into normal real number
            pdf = pdf * math.exp(exponent)

            probability_densities[classification] = pdf

        # Choose a classification based on highest probability
        likely_classification = max(probability_densities, key=probability_densities.get)

        # Aggregate the probability densities for later analysis
        all_probability_densities.append(probability_densities)

        # Update the error rate calculation
        if training_classification + '-total' in error_rate:
            error_rate[training_classification + '-total'] += 1
        else:
            error_rate[training_classification + '-total'] = 1
            error_rate[training_classification + '-incorrect'] = 0

        if likely_classification != training_classification:
            error_rate[training_classification + '-incorrect'] += 1

# Calculate the Error Rate
total = 0.0
incorrect = 0.0
for classification in iris.CLASSIFICATIONS:
    total += error_rate[classification + '-total']
    incorrect += error_rate[classification + '-incorrect']

print 'LDA Training Data Error Rate: ' + str(incorrect/total)
