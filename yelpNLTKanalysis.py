import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
# nltk.download('all')
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve, GridSearchCV
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

pd.set_option('display.max_columns', 10)
data = pd.read_csv('yelp.csv')
print(data.head())
print(data.columns)
print(data.dtypes)
print(data.describe(include='all'))
data['length'] = data['text'].apply(len)
x = data['text'][0:100]
y = data['stars'][0:100]


def text_process(review):
    nopunc = [word for word in review if word not in string.punctuation]
    nopunc = ''.join(nopunc)
    nonums = [word for word in nopunc if (word.isalpha()) | (word ==' ')]
    nonums = ''.join(nonums)
    stop_words_english = stopwords.words('english')
    stop_words_english.extend(['use', 'us', 'make', 'also', 'tell'])
    no_stop_words = [word.lower() for word in nonums.split() if word.lower() not in stop_words_english]
    porter = PorterStemmer()
    stems = []
    for word in no_stop_words:
        stems.append(porter.stem(word))
    return stems


vector = CountVectorizer(analyzer=text_process).fit(x)
X = vector.transform(x)
print(X.toarray())
print("Shape of the sparse matrix: ", X.shape)
print("Non-Zero occurences: ", X.nnz)
# DENSITY OF THE MATRIX
density = (X.nnz/(X.shape[0]*X.shape[1]))*100
print("Density of the matrix = ", density)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,predmnb))
print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
print("Classification Report:",classification_report(y_test,predmnb))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
preddt = dt.predict(x_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,preddt))
print("Score:",round(accuracy_score(y_test,preddt)*100,2))
print("Classification Report:",classification_report(y_test,preddt))


# K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
predknn = knn.predict(x_test)
print("Confusion Matrix for K Neighbors Classifier:")
print(confusion_matrix(y_test,predknn))
print("Score: ",round(accuracy_score(y_test,predknn)*100,2))
print("Classification Report:")
print(classification_report(y_test,predknn))


# XGBoost Classifier
# import xgboost
# from xgboost import XGBClassifier
# xgb = XGBClassifier()
# xgb.fit(x_train,y_train)
# predxgb = xgb.predict(x_test)
# print("Confusion Matrix for XGBoost Classifier:")
# print(confusion_matrix(y_test,predxgb))
# print("Score: ",round(accuracy_score(y_test,predxgb)*100,2))
# print("Classification Report:")
# print(classification_report(y_test,predxgb))


# POSITIVE REVIEW
pr = data['text'][0]
print(pr)
print("Actual Rating: ",data['stars'][0])
pr_t = vector.transform([pr])
print("Predicted Rating:")
print(knn.predict(pr_t)[0])


