import pickle
import numpy as np
import pandas as pd
from abc import ABC

from fake_news_tools import config
from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = config.TEXT_MAIN_PATH + '/ensemble_learning/model/decision_tree_classifier_model.sav'

class DecisionTreeClassifierModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)
    __model = None

    def __init__(self):
        DecisionTreeClassifierModel.__model = pickle.load(open(PATH, 'rb'))

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "Ensemble Learning (Decision Tree Classifier)"

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
        x_temp = x['text'].values.astype('U') # The input is a unicode dataframe of string values
        
        # Classify new records
        preds = DecisionTreeClassifierModel.__model.predict(x_temp)
        value = list(preds)

        # Get probability of classification
        prob_tmp = DecisionTreeClassifierModel.__model.predict_proba(x_temp)
        probability = prob_tmp[0, np.argmax(prob_tmp)] * 100

        return 'Fake' if value[0] else 'Not Fake', probability


