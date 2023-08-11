import pickle
import pandas as pd
from abc import ABC

from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = './model/logistic_regression_model.sav'

class LogisticRegressionModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)
    __model = None

    def __init__(self):
        LogisticRegressionModel.__model = pickle.load(open(PATH, 'rb'))

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "Ensemble Learning (Best Model: Logistic Regression)"

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
        x_temp = x['text'].values.astype('U') # The input is a unicode dataframe of string values

        # Classify new records
        preds = LogisticRegressionModel.__model.predict(x_temp)

        return list(preds)


