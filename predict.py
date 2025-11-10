"""
Quick Predict - Use pre-trained model for single predictions
Useful for API integration or batch processing
"""

import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

def load_model(model_path='./saved_model'):
    """Load the trained model and tokenizer"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first by running: python sms_spam_bert.py"
        )
    
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict(text, model, tokenizer, device, max_length=128):
    """Predict if a message is spam or ham"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    
    return {
        'label': 'spam' if pred == 1 else 'ham',
        'confidence': conf,
        'spam_probability': probs[0][1].item(),
        'ham_probability': probs[0][0].item()
    }

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <message>")
        print("\nExample:")
        print('  python predict.py "WINNER! You have won a prize!"')
        sys.exit(1)
    
    message = ' '.join(sys.argv[1:])
    
    print("Loading model...")
    try:
        model, tokenizer, device = load_model()
        print(f"Model loaded on {device}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Message: {message}")
    print("-" * 60)
    
    result = predict(message, model, tokenizer, device)
    
    print(f"Prediction: {result['label'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Ham:  {result['ham_probability']:.2%}")
    print(f"  Spam: {result['spam_probability']:.2%}")
    
    # Visual indicator
    if result['label'] == 'spam':
        print("\n🚫 This message appears to be SPAM")
    else:
        print("\n✅ This message appears to be legitimate (HAM)")

if __name__ == "__main__":
    main()
