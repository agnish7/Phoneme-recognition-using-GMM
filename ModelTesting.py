import pickle as pk
import numpy as np
from sklearn.mixture  import GaussianMixture
import os

''' For each data point in the test set, the score of each of the GMMs is found and the one with 
the max score is assumed to be the correct model for that data point. This is done for each and every 
data point. The predictions are compared to actual label values from the test set. Based on this a running
accuracy is calculated and printed'''
 
# Enter location of test set files below:

test_data_features_loc = 'Test_data/'+'feature_test_data'+'.p'
test_data_labels_loc = 'Test_data/'+'label_test_data'+'.p'

Y = pk.load(open(test_data_labels_loc, 'rb'))
X = pk.load(open(test_data_features_loc, 'rb'))

#Enter the directory location of the GMMs:

directory = 'GMM_Phonemes'

hits = 0
misses = 0

for i in range(len(X)):
    
    max = -9999999.99999
    max_arg = ''
    
    
    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        GMM = pk.load(open(directory + '/' + filename, 'rb'))
        score = GMM.score(X[i].reshape(1, -1))
        #print(score)
        if score > max:
            max = score
            max_arg = filename
        #print(  str(GMM.score_samples(X[i].reshape(1, -1))) + ': ' + filename   )
        
    
    if str(Y[i]) == str(max_arg)[:-2]:
        hits += 1
    else:
        misses +=1
    print(  str(Y[i]) + ' : ' + str(max_arg)[:-2] + ' : ' + str(hits * 100 / (hits + misses)))

