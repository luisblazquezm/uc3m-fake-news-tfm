import torch
import numpy as np
from abc import ABC

import transformers
from transformers import AutoModel, BertTokenizerFast

import torch
import torch.nn as nn

from fake_news_tools import config
from fake_news_tools.text.models.model_abstraction import ModelAbstraction

PATH = config.TEXT_MAIN_PATH + '/bert/model/c1_fakenews_weights.pt'

class BERT_Arch(nn.Module):

    def __init__(self, bert):  
      super(BERT_Arch, self).__init__()
      self.bert = bert   
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function

    def forward(self, sent_id, mask):           # define the forward pass  
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x

class BertModel(ModelAbstraction, ABC): 
    # Add code for text analysis (Step 1 - NLP)

    __model = None
    __tokenizer = None

    def __init__(self):
        bert = AutoModel.from_pretrained('bert-base-uncased')
        BertModel.__tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        BertModel.__model = BERT_Arch(bert)

        BertModel.__model.load_state_dict(torch.load(PATH)) 

    @staticmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return "BERT"
    
    @staticmethod
    def get_predictions() -> list:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        return ["title"]

    @staticmethod
    def predict(data) -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        
        unseen_news_text = [data]

        # tokenize and encode sequences in the test set
        max_length_content = 15
        tokens_unseen = BertModel.__tokenizer.batch_encode_plus(
            unseen_news_text,
            max_length = max_length_content,
            pad_to_max_length=True,
            truncation=True
        )
        unseen_seq = torch.tensor(tokens_unseen['input_ids'])
        unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

        with torch.no_grad():
            preds = BertModel.__model(unseen_seq, unseen_mask)
            preds = preds.detach().cpu().numpy()

        preds_tmp = np.argmax(preds, axis = 1)
        value = list(preds_tmp)

        # Get probability of classification
        probability = preds[0, preds_tmp] * 100

        return 'Fake' if value[0] else 'Not Fake', probability


