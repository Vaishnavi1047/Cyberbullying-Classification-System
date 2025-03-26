import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from sklearn.metrics import classification_report

class BERT_model:
    def _init_(self, tokenizer="bert-base-uncased", model_name='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2).to(self.device)

def train_model(self, train_dataloader, val_dataloader, epochs=10):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        self.model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')


 def save_model(self,output_dir='cyberbullying_model/'):
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model_state.pt'))
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")


def load_model(self,model_path='cyberbullying_model/'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)        
        # self.model.load_state_dict(torch.load(os.path.join(model_path,'model_state.pt'), map_location=self.device))
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(self.device)
