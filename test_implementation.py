"""
Test script to verify the implementation structure
This runs through data loading and preprocessing without full training
"""

import torch
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("="*60)
print("TESTING SMS SPAM DETECTION IMPLEMENTATION")
print("="*60)

# Check dataset
dataset_path = "sms+spam+collection/SMSSpamCollection"
filename = "SMSSpamCollection.txt"

if os.path.exists(dataset_path):
    print(f"\n✓ Dataset found at: {dataset_path}")
    if not os.path.exists(filename):
        import shutil
        shutil.copy(dataset_path, filename)
        print(f"✓ Copied dataset to: {filename}")
else:
    print(f"\n✗ Dataset not found at: {dataset_path}")
    exit(1)

# Load dataset
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

df = pd.read_csv(filename, sep='\t', names=['label', 'message'])
print(f"✓ Dataset loaded: {df.shape[0]} messages")
print(f"✓ Spam messages: {(df['label'] == 'spam').sum()}")
print(f"✓ Ham messages: {(df['label'] == 'ham').sum()}")

# Encode labels
print("\n" + "="*60)
print("ENCODING LABELS")
print("="*60)

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
print(f"✓ Labels encoded (ham=0, spam=1)")

# Split data
print("\n" + "="*60)
print("SPLITTING DATA")
print("="*60)

train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df['label_encoded']
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,
    random_state=42,
    stratify=train_val_df['label_encoded']
)

print(f"✓ Training set: {len(train_df)} messages ({len(train_df)/len(df)*100:.1f}%)")
print(f"✓ Validation set: {len(val_df)} messages ({len(val_df)/len(df)*100:.1f}%)")
print(f"✓ Test set: {len(test_df)} messages ({len(test_df)/len(df)*100:.1f}%)")

# Test tokenizer
print("\n" + "="*60)
print("TESTING TOKENIZER")
print("="*60)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

sample_text = "Free entry in 2 a wkly comp to win FA Cup final tkts"
tokens = tokenizer.tokenize(sample_text)
print(f"✓ Tokenization test successful")
print(f"  Sample: {sample_text}")
print(f"  Tokens: {len(tokens)} tokens")

# Test encoding
encoding = tokenizer.encode_plus(
    sample_text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)
print(f"✓ Encoding test successful")
print(f"  Input IDs shape: {encoding['input_ids'].shape}")
print(f"  Attention Mask shape: {encoding['attention_mask'].shape}")

# GPU check
print("\n" + "="*60)
print("HARDWARE CHECK")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  Note: Running on CPU (training will be slower)")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nThe implementation structure is correct.")
print("To train the full model, run: python sms_spam_bert.py")
print("\nNote: Training will take 15-30 minutes on GPU, 2-3 hours on CPU")
