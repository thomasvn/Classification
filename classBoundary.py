import qda
import lda
import os


############################################################
#                      Description
############################################################
# To determine whether the classifications were linearly separable, we want to compare the probability densities of all
# classifications for every iris instance we test


############################################################
#                         Analysis
############################################################
# Remove the file if it exists
try:
    os.remove('data/prob.data')
except OSError:
    pass

# Create the pointer to the file
f = open('data/prob.data', 'w+')

# Append all data points to the file
f.write('-------------------- QDA Probability Densities --------------------\n')
for instance in qda.all_probability_densities:
    f.write(str(instance))
    f.write('\n')

f.write('\n\n\n\n\n')

f.write('-------------------- LDA Probability Densities --------------------\n')
for instance in lda.all_probability_densities:
    f.write(str(instance))
    f.write('\n')
