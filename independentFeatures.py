import iris
import qda
import lda
import math
import numpy



############################################################
#                          Cleaning
############################################################


# Format of qda_sigma_square_matrices dictionary
# { 'Iris-virginica': array([[0.45624375, 0.        , 0.        , 0.        ],
#        [0.        , 0.1104    , 0.        , 0.        ],
#        [0.        , 0.        , 0.33669375, 0.        ],
#        [0.        , 0.        , 0.        , 0.0724    ]]),
#   'Iris-setosa': array([[0.2836625 , 0.        , 0.        , 0.        ],
#        [0.        , 0.11596667, 0.        , 0.        ],
#        [0.        , 0.        , 0.18799375, 0.        ],
#        [0.        , 0.        , 0.        , 0.04136458]]),
#   'Iris-versicolor': array([[0.2669    , 0.        , 0.        , 0.        ],
#        [0.        , 0.1081    , 0.        , 0.        ],
#        [0.        , 0.        , 0.19844375, 0.        ],
#        [0.        , 0.        , 0.        , 0.042     ]])
# }
qda_sigma_square_matrices = {}
lda_sigma_square_matrix = numpy.diag(numpy.diag(lda.preserved_sigma_matrix))

for classification in iris.CLASSIFICATIONS:
    qda_sigma_square_matrices[classification] = numpy.diag(numpy.diag(qda.preserved_sigma_matrices[classification]))


############################################################
#                QDA Testing the Testing Set
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
            sigma = qda_sigma_square_matrices[classification]

            instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
            pdf = 1 / (math.sqrt(numpy.linalg.det(sigma)))
            exponent = ((-0.5) * (instance_matrix - mu).transpose() * numpy.linalg.inv(sigma) * (instance_matrix - mu))
            exponent = exponent.item(0)  # Turn scalar into normal real number
            pdf = pdf * math.exp(exponent)

            probability_densities[classification] = pdf

        # Choose a classification based on highest probability
        likely_classification = max(probability_densities, key=probability_densities.get)

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

print 'QDA (Independent Features) Testing Data Error Rate: ' + str(incorrect/total)


############################################################
#               QDA Testing the Training Set
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
            sigma = qda_sigma_square_matrices[classification]

            instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
            pdf = 1 / (math.sqrt(numpy.linalg.det(sigma)))
            exponent = ((-0.5) * (instance_matrix - mu).transpose() * numpy.linalg.inv(sigma) * (instance_matrix - mu))
            exponent = exponent.item(0)  # Turn scalar into normal real number
            pdf = pdf * math.exp(exponent)

            probability_densities[classification] = pdf

        # Choose a classification based on highest probability
        likely_classification = max(probability_densities, key=probability_densities.get)

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

print 'QDA (Independent Features) Training Data Error Rate: ' + str(incorrect/total)


############################################################
#                LDA Testing the Testing Set
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
            sigma = lda_sigma_square_matrix

            instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
            pdf = 1 / (math.sqrt(numpy.linalg.det(sigma)))
            exponent = ((-0.5) * (instance_matrix - mu).transpose() * numpy.linalg.inv(sigma) * (instance_matrix - mu))
            exponent = exponent.item(0)  # Turn scalar into normal real number
            pdf = pdf * math.exp(exponent)

            probability_densities[classification] = pdf

        # Choose a classification based on highest probability
        likely_classification = max(probability_densities, key=probability_densities.get)

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

print 'LDA (Independent Features) Testing Data Error Rate: ' + str(incorrect/total)


############################################################
#                LDA Testing the Training Set
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
            sigma = lda_sigma_square_matrix

            instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
            pdf = 1 / (math.sqrt(numpy.linalg.det(sigma)))
            exponent = ((-0.5) * (instance_matrix - mu).transpose() * numpy.linalg.inv(sigma) * (instance_matrix - mu))
            exponent = exponent.item(0)  # Turn scalar into normal real number
            pdf = pdf * math.exp(exponent)

            probability_densities[classification] = pdf

        # Choose a classification based on highest probability
        likely_classification = max(probability_densities, key=probability_densities.get)

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

print 'LDA (Independent Features) Training Data Error Rate: ' + str(incorrect/total)
