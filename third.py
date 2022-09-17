import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
df = pd.read_csv("preprocessed.csv")
#dataframe = pd.read_csv("training.1600000.processed.noemoticon.csv")
#dataframe.info
    
import pickle
Y = pickle.load(open("Preprocessed_data_sentiment","rb"))
#Y_actual = dataframe.iloc[:, 0].values
#Y_actual = np.array(Y_actual.reshape(-1,1))
#Y_actual = Y_actual[1:]

Y_actual = np.array(Y[:-1])


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
Y_actual = Y_actual.reshape(len(Y_actual), 1)
Y = onehot_encoder.fit_transform(Y_actual)
Y = Y[:,1]
df = np.array(df)
df = df.reshape((len(df),1,300))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size = 0.25, random_state = 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
classifier = Sequential()
classifier.add(LSTM(300, input_shape = (1,300)))
classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 150, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 75, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train , batch_size = 10, epochs = 10)

#classifier = tensorflow.keras.models.load_model("model2")

Y_pred = classifier.predict(X_test)

for i in range(len(Y_pred)):    
    if Y_pred[i]<0.5:
        Y_pred[i]=0
    else:
        Y_pred[i] = 1
        
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)


#classifier.save("model3")

        
