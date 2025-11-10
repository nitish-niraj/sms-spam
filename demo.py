"""
SMS Spam Detection - Quick Demo
A simple script to demonstrate the SMS spam detection system
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

def predict_spam(text, model, tokenizer, device, max_length=128):
    """
    Predict if a message is spam or ham
    
    Args:
        text: SMS message to classify
        model: Trained BERT model
        tokenizer: BERT tokenizer
        device: Device to run on (CPU/GPU)
        max_length: Maximum sequence length
    
    Returns:
        prediction: 'spam' or 'ham'
        confidence: Confidence score
    """
    model.eval()
    
    # Tokenize
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
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = 'spam' if prediction == 1 else 'ham'
    return label, confidence

def main():
    print("="*60)
    print("SMS SPAM DETECTION - QUICK DEMO")
    print("="*60)
    
    # Check if model exists
    model_path = './saved_model'
    if not os.path.exists(model_path):
        print("\nERROR: Model not found!")
        print("Please run 'python sms_spam_bert.py' first to train the model.")
        return
    
    print("\nLoading model...")
    
    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model loaded successfully on {device}!")
    
    # Test messages
    test_messages = [
        "Hey, how are you doing today?",
        "WINNER! You have won a $1000 prize. Call now!",
        "Can we meet for lunch tomorrow?",
        "FREE entry in 2 a wkly comp to win FA Cup final tickets",
        "I'll be there in 5 minutes",
        "Congratulations! You've been selected for a FREE vacation package. Click here now!",
        "Thanks for the meeting today. Let's catch up next week.",
        "URGENT: Your account will be closed. Verify now at bit.ly/fake123",
        "Don't forget to pick up milk on your way home",
        "You have won £1000 cash prize! Text WIN to 12345"
    ]
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    for i, msg in enumerate(test_messages, 1):
        label, confidence = predict_spam(msg, model, tokenizer, device)
        
        # Color coding for terminal
        status = "🚫 SPAM" if label == 'spam' else "✅ HAM"
        
        print(f"\n{i}. Message: {msg}")
        print(f"   Prediction: {status} (confidence: {confidence:.4f})")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter your own messages to check if they're spam!")
    print("(Type 'quit' to exit)\n")
    
    while True:
        user_input = input("Enter a message: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using SMS Spam Detection!")
            break
        
        if not user_input:
            print("Please enter a valid message.\n")
            continue
        
        label, confidence = predict_spam(user_input, model, tokenizer, device)
        status = "🚫 SPAM" if label == 'spam' else "✅ HAM"
        
        print(f"Prediction: {status} (confidence: {confidence:.4f})\n")

if __name__ == "__main__":
    main()
