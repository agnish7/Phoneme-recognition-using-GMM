import numpy as np
import matplotlib as plt
from sklearn.mixture import GaussianMixture
import pickle as pk
import os
import pandas as pd
import random
from numba.types import uint8

''' To split the data intro training and test sets and to generate the GMMs from training data'''

#Set the variable to the location where you have your speech data in .hdf format

data_path = "timit.hdf"

timit_data = pd.read_hdf(data_path)
timit_data.head()

features = np.array(timit_data["features"].tolist())
labels = np.array(timit_data["labels"].tolist())

print(len(features))
print(len(labels))

features_train = np.array(timit_data["features"].tolist())
features_test = np.array(timit_data["features"].tolist())

labels_train = np.array(timit_data["labels"].tolist())
labels_test = np.array(timit_data["labels"].tolist())



i = 0
j = 0

for k in range(len(features)):
    if k % 5 == 0:
        labels_test[i] = labels[k]
        features_test[i] = features[k]
        i+=1
    else:
        labels_train[i] = labels[k]
        features_train[i] = features[k]
        j+=1

ftrain = features_train[0:j:1]
ltrain = labels_train[0:j:1]  

ftest = features_test[0:i:1]
ltest = labels_test[0:i:1]

GMM = GaussianMixture(n_components=64,covariance_type='full',max_iter=150,n_init=1,init_params='kmeans',verbose=0,verbose_interval=10)
for label in ltrain:
    if os.path.isfile('./GMM_Phonemes/'+label+'.p'):
        continue
    GMM.fit(ftrain[ltrain == label])
    file = open('GMM_Phonemes/'+label+'.p','wb')
    pk.dump(GMM,file)
    file.close()


file = open('Test_data/'+'feature_test_data'+'.p','wb')
pk.dump(ftest, file)
file.close

file = open('Test_data/'+'label_test_data'+'.p','wb')
pk.dump(ltest, file)
file.close
