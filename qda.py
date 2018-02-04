import iris
import math
import numpy


# Format of cleaned_iris_data dictionary
# {
#     'Iris-virginica': '6.3,3.3,6.0,2.5;5.8,2.7,5.1,1.9;...',
#     'Iris-setosa': '5.1,3.5,1.4,0.2;4.9,3.0,1.4,0.2;...',
#     'Iris-versicolor': '7.0,3.2,4.7,1.4;6.4,3.2,4.5,1.5;...'
# }
cleaned_iris_data = {}
cleaned_iris_test_data = {}

# Format of mu_matrices dictionary
# Note: All rows of these matrices represent the average of the features of the iris
# {
#     'Iris-virginica': matrix([[6.6225],
#         [2.96  ],
#         [5.6075],
#         [1.99  ]]),
#     'Iris-setosa': matrix([[5.0375],
#         [3.44  ],
#         [1.4625],
#         [0.2325]]),
#     'Iris-versicolor': matrix([[6.01  ],
#         [2.78  ],
#         [4.3175],
#         [1.35  ]])
#  }
mu_matrices = {}

# Format of sigma_matrices dictionary
# Note:
# {
#     'Iris-virginica': matrix([[0.45624375, 0.10765   , 0.34883125, 0.049975  ],
#         [0.10765   , 0.1104    , 0.07905   , 0.0451    ],
#         [0.34883125, 0.07905   , 0.33669375, 0.057825  ],
#         [0.049975  , 0.0451    , 0.057825  , 0.0724    ]]),
#     'Iris-setosa': matrix([[0.12784375, 0.0965    , 0.01265625, 0.01328125],
#         [0.0965    , 0.1294    , 0.002     , 0.0142    ],
#         [0.01265625, 0.002     , 0.02884375, 0.00446875],
#         [0.01328125, 0.0142    , 0.00446875, 0.00969375]]),
#     'Iris-versicolor': matrix([[0.2669    , 0.08445   , 0.167825  , 0.051     ],
#         [0.08445   , 0.1081    , 0.07885   , 0.04425   ],
#         [0.167825  , 0.07885   , 0.19844375, 0.071875  ],
#         [0.051     , 0.04425   , 0.071875  , 0.042     ]])
# }
sigma_matrices = {}
preserved_sigma_matrices = {}  # Assign this value right after training

all_probability_densities = []


############################################################
#                          Cleaning
############################################################

# Clean the data and prepare for calculation of parameters for each classification
for classification in iris.CLASSIFICATIONS:
    for iris_instance in iris.training[classification]:
        iris_data = iris_instance.split(',')
        iris_data = iris_data[:-1]  # The last value of given data is the classification type
        iris_data_string = ','.join(iris_data)
        if classification in cleaned_iris_data:
            cleaned_iris_data[classification] += (iris_data_string + ';')
        else:
            cleaned_iris_data[classification] = (iris_data_string + ';')
    cleaned_iris_data[classification] = cleaned_iris_data[classification][:-1]  # Removes last semicolon in string

    for iris_instance in iris.testing[classification]:
        iris_data = iris_instance.split(',')
        iris_data = iris_data[:-1]
        iris_data_string = ','.join(iris_data)
        if classification in cleaned_iris_test_data:
            cleaned_iris_test_data[classification] += (iris_data_string + ';')
        else:
            cleaned_iris_test_data[classification] = (iris_data_string + ';')
    cleaned_iris_test_data[classification] = cleaned_iris_test_data[classification][:-1]


############################################################
#                          Training
############################################################

# Calculate Mu (mean) of each classification
for classification in iris.CLASSIFICATIONS:
    # Turn cleaned_iris_data into list of matrices
    instance_matrices = []
    iris_data = cleaned_iris_data[classification]
    iris_instances = iris_data.split(';')
    for instance in iris_instances:
        instance_matrices.append(numpy.matrix(instance)) # '1,2,3,4'

    # Add all matrices of same classification together
    for instance in instance_matrices:
        if classification in mu_matrices:
            mu_matrices[classification] += instance
        else:
            mu_matrices[classification] = instance

    # Divide matrix by the number of instances used in training
    mu_matrices[classification] /= (iris.CLASSIFICATION_COUNT[classification] * 0.8)

    # Convention is to have mu as a column vector
    mu_matrices[classification] = mu_matrices[classification].transpose()

# Calculate Sigma (covariance) of each classification
for classification in iris.CLASSIFICATIONS:
    # Turn cleaned_iris_data into list of matrices
    instance_matrices = []
    iris_data = cleaned_iris_data[classification]
    iris_instances = iris_data.split(';')
    for instance in iris_instances:
        instance_matrix = numpy.matrix(instance).transpose()  # Transpose to turn into column vector
        instance_matrix -= mu_matrices[classification]  # Subtract Mu
        instance_matrices.append(numpy.matrix(instance_matrix))

    # Multiply instance_matrices by their transposes then sum all together
    for instance in instance_matrices:
        instance = instance * instance.transpose()  # Outer Product
        if classification in sigma_matrices:
            sigma_matrices[classification] += instance
        else:
            sigma_matrices[classification] = instance

    # Divide matrix by the number of instances used in training
    sigma_matrices[classification] /= (iris.CLASSIFICATION_COUNT[classification] * 0.8)

    preserved_sigma_matrices = sigma_matrices


############################################################
#                          Testing
############################################################

error_rate = {}

for testing_classification in iris.CLASSIFICATIONS:
    iris_test_data = cleaned_iris_test_data[testing_classification]
    iris_instances = iris_test_data.split(';')

    # Plug the instances into each classification's probability density function
    for instance in iris_instances:
        probability_densities = {}
        for classification in iris.CLASSIFICATIONS:
            mu = mu_matrices[classification]
            sigma = sigma_matrices[classification]

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

print 'QDA Testing Data Error Rate: ' + str(incorrect/total)


############################################################
#                  Testing the Training Set
############################################################

error_rate = {}

for training_classification in iris.CLASSIFICATIONS:
    iris_test_data = cleaned_iris_data[training_classification]
    iris_instances = iris_test_data.split(';')

    # Plug the instances into each classification's probability density function
    for instance in iris_instances:
        probability_densities = {}
        for classification in iris.CLASSIFICATIONS:
            mu = mu_matrices[classification]
            sigma = sigma_matrices[classification]

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

print 'QDA Training Data Error Rate: ' + str(incorrect/total)

