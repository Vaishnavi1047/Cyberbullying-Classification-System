import sys
import os

# Get the absolute path of the root directory (Cyber_bullying/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR) 

from utils import *
from Models.BERT_model_pipeline import *
from Dataset.cyber_dataset import *

model = BERT_model()
df = Load_Dataset('Dataset/final_dataset_hinglish.csv', model.tokenizer)
df.prepare_data()
train_dataloader, val_dataloader = df.get_loaders()
model.train_model(train_dataloader, val_dataloader) #we can add more epochs by specifying epochs=5 etc
model.save_model()


'''
Model Training Characteristics Include:
    Epoch 1:
        Average training loss: 0.2008
        Average validation loss: 0.1463

        Validation Classification Report:
                    precision    recall  f1-score   support

        Not Bullying       0.96      0.89      0.93      1298
            Bullying       0.94      0.98      0.96      2332

            accuracy                           0.95      3630
        macro avg       0.95      0.94      0.94      3630
        weighted avg       0.95      0.95      0.95      3630

        Saved best model!

    Epoch 2:
        Average training loss: 0.1146
        Average validation loss: 0.1485

        Validation Classification Report:
                    precision    recall  f1-score   support

        Not Bullying       0.97      0.88      0.92      1298
            Bullying       0.94      0.99      0.96      2332

            accuracy                           0.95      3630
        macro avg       0.95      0.93      0.94      3630
        weighted avg       0.95      0.95      0.95      3630


    Epoch 3:
        Average training loss: 0.0765
        Average validation loss: 0.1505

        Validation Classification Report:
                    precision    recall  f1-score   support

        Not Bullying       0.97      0.90      0.93      1298
            Bullying       0.95      0.98      0.96      2332

            accuracy                           0.95      3630
        macro avg       0.96      0.94      0.95      3630
        weighted avg       0.95      0.95      0.95      3630
'''
