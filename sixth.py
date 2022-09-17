import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
dataframe = pd.read_csv("training.1600000.processed.noemoticon.csv")

def countt():
    countt.count+=1
    print(countt.count)
Y_actual = dataframe.iloc[:, 0].values
Y_actual = np.array(Y_actual.reshape(-1,1))

df = (dataframe[dataframe.columns[-1]])

corpus  = []

Y_actual = np.where(Y_actual==4,1,Y_actual)

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df = df.apply(lambda text: cleaning_stopwords(text))
df.head()

import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df= df.apply(lambda x: cleaning_punctuations(x))
df.tail()


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
df = df.apply(lambda x: cleaning_repeating_char(x))
df.tail()

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
df = df.apply(lambda x: cleaning_numbers(x))
df.tail()

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("\s+", gaps = True)
df = df.apply(tokenizer.tokenize)
df.head()

import nltk
st = nltk.PorterStemmer()
countt.count = 0
def stemming_on_text(data):
    countt()
    text = [st.stem(word) for word in data]
    return text
df= df.apply(lambda x: stemming_on_text(x))
df.head()


lm = nltk.WordNetLemmatizer()
countt.count=0
def lemmatizer_on_text(data):
    countt()
    text = [lm.lemmatize(word) for word in data]
    return text
df = df.apply(lambda x: lemmatizer_on_text(x))
df.head()

countt.count=0
def joining(data):
    countt()
    text = " ".join(data)
    return text
df = df.apply(lambda x: joining(x))
df.head()


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df, Y_actual, test_size = 0.1, random_state = 972364237)


from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression().fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

for i in range(len(Y_pred)):    
    if Y_pred[i]<0.5:
        Y_pred[i]=0
    else:
        Y_pred[i] = 1
        
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)

pickle.dump(classifier, open("model4", "wb"))
pickle.dump(vectoriser, open("model4-vect", "wb"))

