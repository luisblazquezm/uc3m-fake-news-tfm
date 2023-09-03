import re
import numpy as np
from abc import ABC

#### Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fake_news_tools import config
from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = config.TEXT_MAIN_PATH + '/cnn/model/complex_cnn_model_fake_news.h5'
VOCABULARY_SIZE = 200000
SENTENCE_LENGTH = 1000

class CNNModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)

    __tokenizer = None

    def __init__(self):
        CNNModel.__model = load_model(PATH)
        CNNModel.__tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "CNN (Convolutional Neural Network)"
    
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
        cleaned_text = re.sub(r'<.*?>', '', data)
        cleaned_text = re.sub(r'[^\w\s.!?]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        cleaned_text = cleaned_text.lower()

        return cleaned_text
    
    @staticmethod
    def apply_tokenization(data):
        CNNModel.__tokenizer.fit_on_texts(data)
        return CNNModel.__tokenizer.texts_to_sequences(data)
    
    @staticmethod
    def get_embeddings(data):
        return pad_sequences(data, maxlen=SENTENCE_LENGTH)

    @staticmethod
    def predict(data) -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """

        # Apply preprocessing
        data_preprocessed = CNNModel.preprocessing(data=data)

        # Apply tokenization
        tmp_data_preprocessed = [data_preprocessed]
        data_tokenized = CNNModel.apply_tokenization(data=tmp_data_preprocessed)

        # Apply embeddings
        data_embeddings = CNNModel.get_embeddings(data=data_tokenized)
        #final_input = np.array(data_embeddings)

        # Classify new records
        prediction = CNNModel.__model.predict(data_embeddings)
        predicted_class = np.argmax(prediction)

        # Get probability of classification
        probability = np.array(prediction)[0, predicted_class] * 100

        return 'Fake' if int(predicted_class) else 'Not Fake', probability
