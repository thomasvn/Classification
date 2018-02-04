# Initialization of metadata
IRIS_DATA_FILE = 'data/iris.data'
NEW_IRIS_DATA_FILE = 'data/iris4.data'

FEATURE_TO_REMOVE = 3

# Write to a new data file
new_file = open(NEW_IRIS_DATA_FILE, 'w+')

with open(IRIS_DATA_FILE) as f:
    for line in f:
        iris_data = line.split(',')
        del iris_data[FEATURE_TO_REMOVE]
        new_file.write(','.join(iris_data))
