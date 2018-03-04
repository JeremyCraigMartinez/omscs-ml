from os.path import dirname, realpath
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

dir_path = dirname(realpath(__file__))

# uncomment if you do not have stopwords downloaded
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def get_corpus(dataset):
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# Creating the Bag of Words model
def get_X_Y(corpus, dataset):
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    return X, y

# Splitting the dataset into the Training set and Test set
def get_train_test_set():
    # Importing the dataset
    dataset = pd.read_csv(dir_path + '/../../data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    corpus = get_corpus(dataset)
    X, y = get_X_Y(corpus, dataset)

    # X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=0.2)
