"""
SMS Spam Detection with BERT
A complete implementation using state-of-the-art deep learning for text classification
"""

import torch
import transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import urllib.request
import os
import time
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("SMS SPAM DETECTION WITH BERT")
print("="*60)

# ============================================================================
# PHASE 1: ENVIRONMENT SETUP AND VERIFICATION
# ============================================================================

print("\n" + "="*60)
print("PHASE 1: ENVIRONMENT SETUP")
print("="*60)

print("\nLibrary Versions:")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU - training will be slower")

# ============================================================================
# PHASE 2: DATA COLLECTION AND EXPLORATION
# ============================================================================

print("\n" + "="*60)
print("PHASE 2: DATA COLLECTION AND EXPLORATION")
print("="*60)

# Download the dataset if not already present
dataset_path = "sms+spam+collection/SMSSpamCollection"
filename = "SMSSpamCollection.txt"

if os.path.exists(dataset_path):
    print(f"\nDataset found at: {dataset_path}")
    # Copy to working directory for easier access
    if not os.path.exists(filename):
        import shutil
        shutil.copy(dataset_path, filename)
        print(f"Copied dataset to: {filename}")
else:
    print("\nDownloading dataset...")
    url = "https://raw.githubusercontent.com/nitish-niraj/sms-spam/main/sms%2Bspam%2Bcollection/SMSSpamCollection"
    urllib.request.urlretrieve(url, filename)
    print("Dataset downloaded successfully!")

# Load the dataset
df = pd.read_csv(filename, sep='\t', names=['label', 'message'])

print("\nDataset Shape:", df.shape)
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print("\nFirst 5 messages:")
print(df.head())

print("\nLast 5 messages:")
print(df.tail())

print("\nData Types:")
print(df.dtypes)

# Data quality check
print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

print("\nMissing Values:")
print(df.isnull().sum())

duplicates = df.duplicated().sum()
print(f"\nDuplicate messages: {duplicates}")

if duplicates > 0:
    print("Sample duplicate messages:")
    print(df[df.duplicated(keep=False)].head())

# Class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print(df['label'].value_counts())
print("\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100)

# Visualize class distribution
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Distribution of Spam vs Ham Messages', fontsize=14, fontweight='bold')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(df['label'].value_counts()):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
print("\nSaved: class_distribution.png")
plt.close()

# Message length analysis
print("\n" + "="*60)
print("MESSAGE LENGTH ANALYSIS")
print("="*60)

df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

print("\nLength Statistics:")
print(df.groupby('label')[['message_length', 'word_count']].describe())

# Visualize message lengths
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df[df['label'] == 'ham']['message_length'].hist(bins=50, alpha=0.7, 
                                                  label='Ham', ax=axes[0], color='skyblue')
df[df['label'] == 'spam']['message_length'].hist(bins=50, alpha=0.7, 
                                                   label='Spam', ax=axes[0], color='salmon')
axes[0].set_xlabel('Message Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Message Lengths')
axes[0].legend()
axes[0].grid(alpha=0.3)

df[df['label'] == 'ham']['word_count'].hist(bins=50, alpha=0.7, 
                                              label='Ham', ax=axes[1], color='skyblue')
df[df['label'] == 'spam']['word_count'].hist(bins=50, alpha=0.7, 
                                               label='Spam', ax=axes[1], color='salmon')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Word Counts')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('message_length_analysis.png', dpi=300)
print("\nSaved: message_length_analysis.png")
plt.close()

# Sample messages
print("\n" + "="*60)
print("SAMPLE MESSAGES")
print("="*60)

print("\nSample HAM (legitimate) messages:")
for i, msg in enumerate(df[df['label'] == 'ham']['message'].head(3), 1):
    print(f"\n{i}. {msg}")

print("\n\nSample SPAM messages:")
for i, msg in enumerate(df[df['label'] == 'spam']['message'].head(3), 1):
    print(f"\n{i}. {msg}")

# ============================================================================
# PHASE 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*60)
print("PHASE 3: DATA PREPROCESSING")
print("="*60)

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

print("\nLabel Encoding:")
print("ham -> 0")
print("spam -> 1")

print("\nVerification:")
print(df[['label', 'label_encoded']].head())

# Clean text
def clean_text(text):
    """
    Basic text cleaning while preserving meaningful content
    BERT can handle most variations, so we keep cleaning minimal
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['message_clean'] = df['message'].apply(clean_text)

print("\nText Cleaning Examples:")
print("\nOriginal vs Cleaned:")
for idx in [0, 100, 500]:
    print(f"\nOriginal: {df['message'].iloc[idx]}")
    print(f"Cleaned:  {df['message_clean'].iloc[idx]}")

# Split data
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

print("\n" + "="*60)
print("DATA SPLIT SUMMARY")
print("="*60)
print(f"Training set: {len(train_df)} messages ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} messages ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} messages ({len(test_df)/len(df)*100:.1f}%)")

print("\nClass distribution in each set:")
print("\nTraining:")
print(train_df['label'].value_counts())
print("\nValidation:")
print(val_df['label'].value_counts())
print("\nTest:")
print(test_df['label'].value_counts())

# ============================================================================
# PHASE 4: BERT TOKENIZATION
# ============================================================================

print("\n" + "="*60)
print("PHASE 4: BERT TOKENIZATION")
print("="*60)

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("\nTokenizer loaded successfully!")
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Test tokenization
sample_text = "Free entry in 2 a wkly comp to win FA Cup final tkts"
print(f"\nSample text: {sample_text}")

tokens = tokenizer.tokenize(sample_text)
print(f"\nTokens: {tokens}")

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"\nToken IDs: {token_ids}")

encoding = tokenizer.encode_plus(
    sample_text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

print(f"\nInput IDs shape: {encoding['input_ids'].shape}")
print(f"Attention Mask shape: {encoding['attention_mask'].shape}")

# Analyze message lengths for max_length selection
print("\nAnalyzing message lengths for max_length selection...")
message_lengths = []

for message in df['message_clean']:
    tokens = tokenizer.encode(message, add_special_tokens=True)
    message_lengths.append(len(tokens))

df['token_count'] = message_lengths

print("\nToken Count Statistics:")
print(df['token_count'].describe())

# Visualize distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(message_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.title('Distribution of Message Lengths (in tokens)')
plt.axvline(x=128, color='r', linestyle='--', label='max_length=128')
plt.axvline(x=np.percentile(message_lengths, 95), color='g', linestyle='--', 
            label=f'95th percentile={int(np.percentile(message_lengths, 95))}')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
percentiles = [50, 75, 90, 95, 99, 100]
values = [np.percentile(message_lengths, p) for p in percentiles]
plt.bar(range(len(percentiles)), values, tick_label=[f'{p}%' for p in percentiles],
        edgecolor='black', alpha=0.7)
plt.xlabel('Percentile')
plt.ylabel('Token Count')
plt.title('Token Count Percentiles')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('token_length_analysis.png', dpi=300)
print("\nSaved: token_length_analysis.png")
plt.close()

max_length_95 = int(np.percentile(message_lengths, 95))
max_length_99 = int(np.percentile(message_lengths, 99))

print(f"\nRecommended max_length:")
print(f"  - 95% of messages fit in {max_length_95} tokens")
print(f"  - 99% of messages fit in {max_length_99} tokens")
print(f"  - Suggested: 128 tokens (covers most messages efficiently)")

MAX_LENGTH = 128
truncated = (df['token_count'] > MAX_LENGTH).sum()
print(f"\nWith max_length={MAX_LENGTH}:")
print(f"  - {truncated} messages ({truncated/len(df)*100:.2f}%) will be truncated")
print(f"  - {len(df)-truncated} messages ({(len(df)-truncated)/len(df)*100:.2f}%) fit completely")

# Create PyTorch Dataset
class SMSDataset(Dataset):
    """
    Custom Dataset for SMS messages
    Handles tokenization and prepares data for BERT
    """
    
    def __init__(self, messages, labels, tokenizer, max_length=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, idx):
        message = str(self.messages[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SMSDataset(
    messages=train_df['message_clean'].values,
    labels=train_df['label_encoded'].values,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

val_dataset = SMSDataset(
    messages=val_df['message_clean'].values,
    labels=val_df['label_encoded'].values,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

test_dataset = SMSDataset(
    messages=test_df['message_clean'].values,
    labels=test_df['label_encoded'].values,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

print("\n" + "="*60)
print("DATASETS CREATED")
print("="*60)
print(f"Training dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")

print("\nSample from training dataset:")
sample = train_dataset[0]
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Attention Mask shape: {sample['attention_mask'].shape}")
print(f"Label: {sample['labels']}")

# ============================================================================
# PHASE 5: BUILDING AND CONFIGURING BERT MODEL
# ============================================================================

print("\n" + "="*60)
print("PHASE 5: BUILDING AND CONFIGURING BERT MODEL")
print("="*60)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

model = model.to(device)

print("\nModel loaded successfully!")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (float32)")

# Configure training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to='none',
    seed=42
)

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Training batch size: {training_args.per_device_train_batch_size}")
print(f"Evaluation batch size: {training_args.per_device_eval_batch_size}")
print(f"Learning rate warmup steps: {training_args.warmup_steps}")
print(f"Weight decay: {training_args.weight_decay}")
print(f"Mixed precision (FP16): {training_args.fp16}")

total_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
print(f"\nTotal training steps: {total_steps}")
print(f"Evaluation frequency: Every {training_args.logging_steps} steps")

# Define evaluation metrics
def compute_metrics(pred):
    """Compute evaluation metrics during training"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    accuracy = accuracy_score(labels, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)
print("""
Metrics we'll track:
1. Accuracy: Overall correctness
2. Precision: Of predicted spam, how much is actually spam?
3. Recall: Of actual spam, how much did we catch?
4. F1-Score: Harmonic mean of precision and recall
""")

# ============================================================================
# PHASE 6: TRAINING THE BERT MODEL
# ============================================================================

print("\n" + "="*60)
print("PHASE 6: TRAINING THE BERT MODEL")
print("="*60)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\nTrainer initialized!")
print("Ready to start training!")

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print("This may take several minutes...")
print("You'll see progress bars and periodic evaluation results\n")

start_time = time.time()

# Train the model
train_result = trainer.train()

end_time = time.time()
training_time = end_time - start_time

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Total training time: {training_time/60:.2f} minutes")
print(f"Average time per epoch: {training_time/training_args.num_train_epochs/60:.2f} minutes")

train_metrics = train_result.metrics
print("\nFinal Training Metrics:")
for key, value in train_metrics.items():
    print(f"  {key}: {value:.4f}")

# Visualize training progress
print("\n" + "="*60)
print("VISUALIZING TRAINING PROGRESS")
print("="*60)

if os.path.exists('./results/trainer_state.json'):
    with open('./results/trainer_state.json', 'r') as f:
        trainer_state = json.load(f)

    log_history = trainer_state['log_history']

    train_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []
    steps = []

    for entry in log_history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            steps.append(entry['step'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_accuracy.append(entry['eval_accuracy'])
            eval_f1.append(entry['eval_f1'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training Loss
    if train_loss:
        axes[0, 0].plot(steps, train_loss, marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].grid(alpha=0.3)

    # Validation Loss
    if eval_loss:
        epochs = list(range(1, len(eval_loss) + 1))
        axes[0, 1].plot(epochs, eval_loss, marker='s', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss per Epoch')
        axes[0, 1].grid(alpha=0.3)

    # Validation Accuracy
    if eval_accuracy:
        axes[1, 0].plot(epochs, eval_accuracy, marker='s', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy per Epoch')
        axes[1, 0].set_ylim([0.9, 1.0])
        axes[1, 0].grid(alpha=0.3)

    # Validation F1 Score
    if eval_f1:
        axes[1, 1].plot(epochs, eval_f1, marker='s', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score per Epoch')
        axes[1, 1].set_ylim([0.9, 1.0])
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300)
    print("\nSaved: training_progress.png")
    plt.close()

# ============================================================================
# PHASE 7: MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("PHASE 7: MODEL EVALUATION")
print("="*60)

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)

print("\nTest Set Results:")
for key, value in test_results.items():
    if key.startswith('eval_'):
        metric_name = key.replace('eval_', '').upper()
        print(f"  {metric_name}: {value:.4f}")

# Generate predictions
predictions_output = trainer.predict(test_dataset)
predictions = predictions_output.predictions.argmax(-1)
true_labels = predictions_output.label_ids

# Calculate comprehensive metrics
from sklearn.metrics import classification_report

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_labels, predictions, 
                          target_names=['Ham', 'Spam']))

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("\nSaved: confusion_matrix.png")
plt.close()

# Analyze errors
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

test_df_reset = test_df.reset_index(drop=True)
errors = test_df_reset[predictions != true_labels].copy()
errors['predicted'] = predictions[predictions != true_labels]
errors['true_label'] = true_labels[predictions != true_labels]

print(f"\nTotal errors: {len(errors)} out of {len(test_df)} ({len(errors)/len(test_df)*100:.2f}%)")

if len(errors) > 0:
    print("\nFalse Positives (Ham predicted as Spam):")
    fp = errors[errors['true_label'] == 0].head(5)
    for idx, row in fp.iterrows():
        print(f"\n  Message: {row['message']}")
    
    print("\n\nFalse Negatives (Spam predicted as Ham):")
    fn = errors[errors['true_label'] == 1].head(5)
    for idx, row in fn.iterrows():
        print(f"\n  Message: {row['message']}")

# ============================================================================
# PHASE 8: MODEL DEPLOYMENT
# ============================================================================

print("\n" + "="*60)
print("PHASE 8: MODEL DEPLOYMENT")
print("="*60)

# Save model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("\nModel and tokenizer saved to: ./saved_model")

# Create prediction function
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

# Test the prediction function
print("\n" + "="*60)
print("TESTING PREDICTION FUNCTION")
print("="*60)

test_messages = [
    "Hey, how are you doing today?",
    "WINNER! You have won a $1000 prize. Call now!",
    "Can we meet for lunch tomorrow?",
    "FREE entry in 2 a wkly comp to win FA Cup final tickets",
    "I'll be there in 5 minutes"
]

print("\nSample Predictions:")
for msg in test_messages:
    label, confidence = predict_spam(msg, model, tokenizer, device)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {label.upper()} (confidence: {confidence:.4f})")

# Save summary
summary = {
    'model': 'bert-base-uncased',
    'total_messages': len(df),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'test_accuracy': test_results['eval_accuracy'],
    'test_precision': test_results['eval_precision'],
    'test_recall': test_results['eval_recall'],
    'test_f1': test_results['eval_f1'],
    'training_time_minutes': training_time/60,
}

with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("PROJECT COMPLETED!")
print("="*60)
print("\nGenerated Files:")
print("  - class_distribution.png")
print("  - message_length_analysis.png")
print("  - token_length_analysis.png")
print("  - training_progress.png")
print("  - confusion_matrix.png")
print("  - saved_model/ (directory with model and tokenizer)")
print("  - model_summary.json")

print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("Thank you for using SMS Spam Detection with BERT!")
print("="*60)
