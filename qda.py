import iris
import numpy

# Estimate the parameters
# Mew = The mean of all different classifications
#       Each classification will have a 4x1 matrix
#       Place this inside NumPy's matrix representation
# Sigma = Outer Product

cleaned_iris_data = {
    'Iris-setosa': '',
    'Iris-versicolor': '',
    'Iris-virginica': ''
}

mew_matrices = {}
sigma_matrices = {}

# Clean the data and prepare for calculation of parameters for each classification
for classification in iris.CLASSIFICATIONS:
    for iris_instance in iris.training[classification]:
        iris_data = iris_instance.split(',')  # Separates string by comma
        iris_data = iris_data[:-1]  # Take all values except for last value
        iris_data_string = ','.join(iris_data)  # Turn list back into comma separated string
        cleaned_iris_data[classification] += iris_data_string
        cleaned_iris_data[classification] += ';'
    cleaned_iris_data[classification] = cleaned_iris_data[classification][:-1]  # Removes last semicolon in string


# Calculate mew of each classification
instance_matrices = []
for classification in iris.CLASSIFICATIONS:
    iris_data = cleaned_iris_data[classification]
    iris_instances = iris_data.split(';')
    for instance in iris_instances:
        instance_matrices.append(numpy.matrix(instance))
    break
    # TODO: Add all matrices inside 'instance_matrices' together
    # TODO: Divide that single matrix by N

print instance_matrices
