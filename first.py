import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
dataframe = pd.read_csv("training.1600000.processed.noemoticon.csv")
dataframe = dataframe.sample(50001)
Y_actual = dataframe.iloc[:, 0].values
Y_actual = np.array(Y_actual.reshape(-1,1))

df = (dataframe[dataframe.columns[-1]])

wordnet_lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
def stemlemm(a):
    a = porter.stem(a)
    a = wordnet_lemmatizer.lemmatize(a)
    return a
    
    
df = np.array(df)
corpus  = []
for i in range(0, len(df)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', df[i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize    
corpus_temp  = []
corpus_split  = []
for i in range(len(corpus)):
    print(i)
    temp = corpus[i].split()
    corpus_split.append(temp)


import gensim
#model = gensim.models.Word2Vec(corpus_split, min_count = 1)
model = gensim.models.KeyedVectors.load_word2vec_format('C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\gensim\\models\\GoogleNews-vectors-negative300.bin', binary = True)

#Using goolge's data to convert corpus into vectors with 300 parameters
#for loop: take each row , then take each word in the row and convert, then average 
#of each row per parameter so each tweet has 300 parameter instead of having mult-
#iple words with 300 parameter
corpus_numbers = []
    
for i in range(len(corpus)):
    print(i)
    actual_words = 0    
    average = [0]*300
    for some_word in corpus_split[i]:
        try:
            word  = model[some_word]
        except:
            word = [0]*300
        if word  != "nan":
            actual_words += 1 
            for k in range(300):                
                number_of_terms = len(corpus_split[i])     
                average[k] += word[k]
                    
    for flag  in range(300) :
        if actual_words == 0:
            actual_words = 1
        average [flag] = average [flag]/actual_words
    
    corpus_numbers.append(average)
#convert to np.array   
corpus_final = np.array(corpus_numbers)

#pickle.dump(corpus_final, open('Preprocessed_data','wb'))
pickle.dump(Y_actual, open('Preprocessed_data_sentiment2', 'wb'))

import csv
count = 0
with open('preprocessed2.csv','w',newline="") as result_file:
    wr = csv.writer(result_file)
    for i in corpus_final:
        count+=1
        print(count)
        wr.writerow(i)   
