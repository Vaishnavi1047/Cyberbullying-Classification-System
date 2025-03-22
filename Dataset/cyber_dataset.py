import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset 
from sklearn.model_selection import train_test_split
import emoji

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class CyberbullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
class Load_Dataset:
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.df = pd.read_csv(file_path)
        self.df = self.df.reset_index(drop=True)
        self.df['processed_text'] = self.df['headline'].apply(self.preprocess_text)


    def preprocess_text(self,text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # remove urls
        text = re.sub(r'@\w+', '', text)  # remove mentions
        text = re.sub(r'#\w+', '', text)  # remove hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
        text = emoji.demojize(text) # recognize emojis

        #tokenization
        tokens = word_tokenize(text)

        #remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words] 

        #lemmatization       
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)

    def prepare_data(self):
        self.df['label'] = self.df['label'].map({0: 0, -1: 1})
    
    def get_loaders(self):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.df['processed_text'].values,  
            self.df['label'].values,         
            test_size=0.2,
            random_state=82,
            stratify=self.df['label']
        )
        
        train_dataset = CyberbullyingDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = CyberbullyingDataset(val_texts, val_labels, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16)

        return train_dataloader, val_dataloader

#cyber_dataset.py can be reused from another script that calls cyber_dataset as well as transformers to your existing file path of dataset.