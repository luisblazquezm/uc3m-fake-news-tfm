import pandas as pd
import re
import nltk
import numpy as np
from abc import ABC

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fake_news_tools import config
from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = config.TEXT_MAIN_PATH + '/lstm/model/lstm_model_fake_news.h5'
VOCABULARY_SIZE = 500
SENTENCE_LENGTH = 20

class LSTMModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)

    __stemmer = None
    __model = None

    def __init__(self):
        LSTMModel.__stemmer = PorterStemmer()
        LSTMModel.__model = load_model(PATH)

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "LSTM (Long Short Term Memory)"
    
    @staticmethod
    def get_predictions() -> list:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return ["title", "text"]
    
    @staticmethod
    def preprocessing(data):
        corpus = []
        words = []
        for i in range(0,len(data)):
            review = re.sub('[^a-zA-Z0-9]',' ', data['text'][i])
            review = review.lower()
            review = review.split()
            review = [LSTMModel.__stemmer.stem(word) for word in review if not word in stopwords.words('english')]
            statements = ' '.join(review)
            corpus.append(statements)
            words.append(review)

        return corpus
    
    @staticmethod
    def apply_one_hot_encoding(data):
        return [one_hot(words, VOCABULARY_SIZE) for words in data] 
    
    @staticmethod
    def get_embeddings(data):
        return pad_sequences(data, padding='post', maxlen=SENTENCE_LENGTH)

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

        # Apply preprocessing
        data_preprocessed = LSTMModel.preprocessing(data=x)

        # Apply one hot encoding
        data_one_hot = LSTMModel.apply_one_hot_encoding(data=data_preprocessed)

        # Apply embeddings
        data_embeddings = LSTMModel.get_embeddings(data=data_one_hot)
        final_input = np.array(data_embeddings)

        # Classify new records
        preds = (LSTMModel.__model.predict(final_input) > 0.5).astype("int32")
        value = preds.tolist()

        # Get probability of classification
        probability = np.array(value)[0, 0] * 100

        return 'Fake' if np.array(value)[0, 0] > 0.5 else 'Not Fake', probability

