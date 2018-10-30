import numpy as np
import matplotlib as plt
from sklearn.mixture import GaussianMixture
import pickle as pk
import os
import pandas as pd

data_path = "timit.hdf"
timit_data = pd.read_hdf(data_path)
timit_data.head()

features = np.array(timit_data["features"].tolist())
labels = np.array(timit_data["labels"].tolist())

GMM = GaussianMixture(n_components=64,covariance_type='full',max_iter=150,n_init=1,init_params='kmeans',verbose=0,verbose_interval=10)

for label in labels:
    if os.path.isfile('./GMM_Phonemes/'+label+'.p'):
        continue
    GMM.fit(features[labels == label])
    file = open('GMM_Phonemes/'+label+'.p','wb')
    pk.dump(GMM,file)
    file.close()
   
