import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from sklearn.metrics import classification_report

class BERT_model:
    def _init_(self, tokenizer="bert-base-uncased", model_name='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2).to(self.device)
