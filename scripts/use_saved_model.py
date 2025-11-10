"""Load and use the saved trained model for predictions"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load the saved model and tokenizer
MODEL_DIR = "saved_model"

print("Loading saved model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Determine device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}\n")

# Create a pipeline for easy inference
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)

# Test messages
test_messages = [
    "You won a free iPhone! Click here to claim.",
    "Hey, are we still on for dinner tonight?",
    "URGENT: Your account has been compromised. Verify now!",
    "Sure, see you at 5pm tomorrow",
    "Congratulations! You have been selected to win $1000!"
]

print("="*60)
print("TESTING SAVED MODEL")
print("="*60 + "\n")

for msg in test_messages:
    result = classifier(msg)
    label = result[0]['label']
    score = result[0]['score']
    print(f"Message: {msg}")
    print(f"Prediction: {label.upper()} (confidence: {score:.2%})\n")

print("="*60)
print("✅ Model inference complete!")
print("="*60)
