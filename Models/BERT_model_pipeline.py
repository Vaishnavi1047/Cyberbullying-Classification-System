import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from sklearn.metrics import classification_report
class BERT_model:
    def _init_(self, tokenizer="bert-base-uncased", model_name='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2).to(self.device)


 def save_model(self,output_dir='cyberbullying_model/'):
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model_state.pt'))
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model and tokenizer saved to {output_dir}")
