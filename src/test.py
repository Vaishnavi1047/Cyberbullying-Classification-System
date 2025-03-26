from utils import *
from Models.BERT_model_pipeline import *
'''
Save the model in the root directory under a folder named cyberbullying_model
'''

Cybermodel = BERT_model().load_model()

test_messages = [
    "You're such a loser, nobody likes you!",
    "I hope you die in your sleep, worthless piece of garbage",
    "You're too stupid to understand anything",
    "Have a great day! Looking forward to our meeting",
    "Nice work on the project yesterday"
]

Cybermodel.model.eval()
for message in test_messages:
    encoded = Cybermodel.tokenizer(
        message,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(Cybermodel.device)
    
    with torch.no_grad():
        outputs = Cybermodel.model(**encoded)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    print("\nMessage:", message)
    print("Classification:", "Not Bullying" if predicted_class == 0 else "Bullying")
    print(f"Confidence: {confidence:.2%}")


'''
Upon testing following output was achieved:

Message: You're such a loser, nobody likes you!
Classification: Bullying
Confidence: 86.59%

Message: I hope you die in your sleep, worthless piece of garbage
Classification: Bullying
Confidence: 97.75%

Message: You're too stupid to understand anything
Classification: Bullying
Confidence: 59.11%

Message: Have a great day! Looking forward to our meeting
Classification: Not Bullying
Confidence: 99.66%

Message: Nice work on the project yesterday
Classification: Not Bullying
Confidence: 99.61%
'''
