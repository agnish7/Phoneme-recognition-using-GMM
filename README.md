# Phoneme-recognition-using-GMM

1. Change the paths to directories wherever required.
2. Accuracy is calculated by running each test file through all of the models (generated by the modeltrainer file) 
and assigning the label of the model that produces the highest score to it.
3. If the label is correct then the variable 'hits' is increased by 1. If not then 'misses' is increased by 1.
4. A running accuracy is calculated as hits / (hits + misses). This is displayed on the terminal. The last accuracy reading 
will indicate the overall accuracy.
