# Initialization of metadata
IRIS_DATA_FILE = 'data/iris4.data'
CLASSIFICATIONS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
CLASSIFICATION_COUNT = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}

# Initialization of training and testing lists
# Format of training and testing dictionaries
# {
#   'Iris-setosa': ['_,_,_,Iris-setosa', ...]
#   'Iris-versicolor': ['_,_,_,Iris-versicolor', ...]
#   'Iris-verginica': ['_,_,_,Iris-verginica', ...]
# }
training = {
    'Iris-setosa': [],
    'Iris-versicolor': [],
    'Iris-virginica': []
}
testing = {
    'Iris-setosa': [],
    'Iris-versicolor': [],
    'Iris-virginica': []
}

# Read from data file
iris_file_object = open(IRIS_DATA_FILE, 'r')
iris_data = iris_file_object.readlines()

# Counts the number of each classification
for iris_instance in iris_data:
    for classification in CLASSIFICATIONS:
        if classification in iris_instance:
            CLASSIFICATION_COUNT[classification] += 1

# Separates data into training and testing
for iris_instance in iris_data:
    for classification in CLASSIFICATIONS:
        if classification in iris_instance:
            if len(training[classification]) < (CLASSIFICATION_COUNT[classification] * 0.8):
                training[classification].append(iris_instance)
            else:
                testing[classification].append(iris_instance)
