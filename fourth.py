import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("preprocessed2.csv")
#dataframe = pd.read_csv("training.1600000.processed.noemoticon.csv")
#dataframe.info
      
import pickle
Y = pickle.load(open("Preprocessed_data_sentiment2","rb"))  

Y_actual = np.array(Y[:-1])

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
Y_actual = Y_actual.reshape(len(Y_actual), 1)
Y = onehot_encoder.fit_transform(Y_actual)
Y = Y[:,1]
df = np.array(df)
#df = df.reshape((1599998,1,300))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)   

#classifier.save("model2")

Y_pred = classifier.predict(X_test)

for i in range(len(Y_pred)):    
    if Y_pred[i]<0.5:
        Y_pred[i]=0
    else:
        Y_pred[i] = 1
        
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)
