import pickle
import re
import nltk
import numpy as np
import pandas as pd
from abc import ABC

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

from fake_news_tools import config
from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = config.TEXT_MAIN_PATH + '/feature_engineering/model/passive_aggresive_model_tfidf.sav'
MAX_FEATURES_VECTORIZER= 500

class PassiveAgressiveTFIDFModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)
    __stemmer = None
    __model = None

    def __init__(self):
        PassiveAgressiveTFIDFModel.__stemmer = PorterStemmer()
        PassiveAgressiveTFIDFModel.__model = pickle.load(open(PATH, 'rb'))

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "Feature Engineering (TFIDF + Passive Aggresive algorithm)"
    
    @staticmethod
    def preprocessing(data):
        corpus = []
        words = []
        for i in range(0,len(data)):
            review = re.sub('[^a-zA-Z0-9]',' ', data[i])
            review = review.lower()
            review = review.split()
            review = [PassiveAgressiveTFIDFModel.__stemmer.stem(word) for word in review if not word in stopwords.words('english')]
            statements = ' '.join(review)
            corpus.append(statements)
            words.append(review)

        return corpus
    
    @staticmethod
    def get_predictions() -> list:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return ["title", "text"]

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

        data_preprocessed = PassiveAgressiveTFIDFModel.preprocessing(data=data)

        # Prepare vectorizers (CountVectorizer)
        tfidf_handler_test = TfidfVectorizer(max_features=MAX_FEATURES_VECTORIZER,ngram_range=(1,3))
        data_tfidf_vectorizer = tfidf_handler_test.fit_transform(data_preprocessed).toarray()

        # Classify new records
        preds = PassiveAgressiveTFIDFModel.__model.predict(data_tfidf_vectorizer)
        value = list(preds)

        return 'Fake' if value[0] else 'Not Fake', 0

