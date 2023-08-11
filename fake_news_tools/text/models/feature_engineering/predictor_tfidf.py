import pickle
import re
import nltk
import numpy as np
import pandas as pd
from abc import ABC

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = './model/passive_aggresive_model_count.sav'
MAX_FEATURES_VECTORIZER= 500

class PassiveAgressiveCountModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)
    __stemmer = None
    __model = None

    def __init__(self):
        PassiveAgressiveCountModel.__stemmer = PorterStemmer()
        PassiveAgressiveCountModel.__model = pickle.load(open(PATH, 'rb'))

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "Feature Engineering (CountVectorizer + Passive Aggresive algorithm)"
    
    @staticmethod
    def preprocessing(data):
        corpus = []
        words = []
        for i in range(0,len(data)):
            review = re.sub('[^a-zA-Z0-9]',' ', data['text'][i])
            review = review.lower()
            review = review.split()
            review = [PassiveAgressiveCountModel.__stemmer.stem(word) for word in review if not word in stopwords.words('english')]
            statements = ' '.join(review)
            corpus.append(statements)
            words.append(review)

        return corpus

    @staticmethod
    def predict(data) -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        # Transform into dataframe
        lst = [data]
        x = pd.DataFrame(lst, index =[0], columns =['text'])
        data = x['text'].values.astype('U') # The input is a unicode dataframe of string values

        data_preprocessed = PassiveAgressiveCountModel.preprocessing(data=data)

        # Prepare vectorizers (CountVectorizer)
        count_vectorizer_handler_test = CountVectorizer(max_features=MAX_FEATURES_VECTORIZER, ngram_range=(1,3))
        data_count_vectorizer = count_vectorizer_handler_test.fit_transform(data_preprocessed).toarray()

        # Classify new records
        preds = PassiveAgressiveCountModel.__model.predict(data_count_vectorizer)

        return list(preds)


