# SMS Spam Detection with BERT - Complete Project Roadmap

## Project Overview

You'll build a cutting-edge SMS spam detection system using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art deep learning model for natural language processing. Your dataset contains 5,574 SMS messages labeled as spam or ham (legitimate).

BERT represents a paradigm shift from traditional machine learning. Instead of using pre-extracted features, BERT learns to understand the meaning and context of text directly, making it exceptionally powerful for text classification tasks.

---

## Phase 1: Understanding BERT and Environment Setup

### Step 1.1: What Makes BERT Special?

Before diving into code, let's understand why BERT is revolutionary:

**Traditional Approaches (like KNN/SVM):**
- Require manual feature extraction (word frequencies, TF-IDF)
- Treat words independently without context
- Limited understanding of semantic meaning

**BERT Approach:**
- Pre-trained on massive text corpora (Wikipedia, books)
- Understands context bidirectionally (looks at words before AND after)
- Captures semantic relationships (knows "bank" means different things in "river bank" vs "bank account")
- Uses attention mechanisms to focus on relevant parts of text

Think of BERT as a person who has read millions of books and learned the nuances of language, versus someone who just counts word frequencies.

### Step 1.2: Install Required Libraries

BERT requires specialized deep learning libraries. Install them carefully:

```bash
# Core deep learning framework
pip install torch torchvision torchaudio

# Hugging Face Transformers - provides pre-trained BERT models
pip install transformers

# Data manipulation and visualization
pip install pandas numpy matplotlib seaborn scikit-learn

# Progress bars for training
pip install tqdm

# For GPU acceleration (if available)
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Important:** BERT is computationally intensive. While it can run on CPU, it will be much faster with a GPU. Google Colab provides free GPU access if you don't have one locally.

### Step 1.3: Verify Installation and Check GPU Availability

Create a new Python file `sms_spam_bert.py` and verify your setup:

```python
import torch
import transformers
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU - training will be slower")
```

---

## Phase 2: Data Collection and Exploration

### Step 2.1: Download the Dataset

You can download the dataset from the GitHub repository. The file is `SMSSpamCollection` (note: it's a tab-separated file, not CSV):

```python
import urllib.request
import os

# Download the dataset if not already present
url = "https://raw.githubusercontent.com/nitish-niraj/sms-spam/main/sms%2Bspam%2Bcollection/SMSSpamCollection"
filename = "SMSSpamCollection.txt"

if not os.path.exists(filename):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Dataset downloaded successfully!")
else:
    print("Dataset already exists.")
```

### Step 2.2: Load and Explore the Dataset

The SMSSpamCollection file has a unique format - it's tab-separated with no header:

```python
# Load the dataset
# Format: label\tmessage
df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['label', 'message'])

print("Dataset Shape:", df.shape)
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Display first few messages
print("\nFirst 5 messages:")
print(df.head())

# Display last few messages
print("\nLast 5 messages:")
print(df.tail())

# Check data types
print("\nData Types:")
print(df.dtypes)
```

### Step 2.3: Comprehensive Data Analysis

Understanding your data is crucial for building an effective model:

```python
print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate messages
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

# Add value labels on bars
for i, v in enumerate(df['label'].value_counts()):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.show()
```

### Step 2.4: Analyze Message Characteristics

Let's understand the differences between spam and ham messages:

```python
print("\n" + "="*60)
print("MESSAGE LENGTH ANALYSIS")
print("="*60)

# Calculate message lengths
df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# Statistics by class
print("\nLength Statistics:")
print(df.groupby('label')[['message_length', 'word_count']].describe())

# Visualize message lengths
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Message character length
df[df['label'] == 'ham']['message_length'].hist(bins=50, alpha=0.7, 
                                                  label='Ham', ax=axes[0], color='skyblue')
df[df['label'] == 'spam']['message_length'].hist(bins=50, alpha=0.7, 
                                                   label='Spam', ax=axes[0], color='salmon')
axes[0].set_xlabel('Message Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Message Lengths')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Word count
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
plt.show()

# Show sample messages
print("\n" + "="*60)
print("SAMPLE MESSAGES")
print("="*60)

print("\nSample HAM (legitimate) messages:")
for i, msg in enumerate(df[df['label'] == 'ham']['message'].head(3), 1):
    print(f"\n{i}. {msg}")

print("\n\nSample SPAM messages:")
for i, msg in enumerate(df[df['label'] == 'spam']['message'].head(3), 1):
    print(f"\n{i}. {msg}")
```

---

## Phase 3: Data Preprocessing for BERT

### Step 3.1: Encode Labels

Convert text labels to numerical format:

```python
from sklearn.preprocessing import LabelEncoder

# Encode labels: ham=0, spam=1
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

print("\nLabel Encoding:")
print("ham -> 0")
print("spam -> 1")

print("\nVerification:")
print(df[['label', 'label_encoded']].head())
```

### Step 3.2: Clean and Prepare Text Data

While BERT is robust to text variations, basic cleaning can help:

```python
import re

def clean_text(text):
    """
    Basic text cleaning while preserving meaningful content
    BERT can handle most variations, so we keep cleaning minimal
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

# Apply cleaning
df['message_clean'] = df['message'].apply(clean_text)

print("\nText Cleaning Examples:")
print("\nOriginal vs Cleaned:")
for idx in [0, 100, 500]:
    print(f"\nOriginal: {df['message'].iloc[idx]}")
    print(f"Cleaned:  {df['message_clean'].iloc[idx]}")
```

### Step 3.3: Split Data into Training, Validation, and Test Sets

For deep learning, we need three sets:
- **Training set** (70%): To train the model
- **Validation set** (15%): To tune hyperparameters and monitor training
- **Test set** (15%): Final evaluation on unseen data

```python
from sklearn.model_selection import train_test_split

# First split: separate test set (15%)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df['label_encoded']
)

# Second split: separate validation set from training (15% of remaining 85% ≈ 12.75% of total)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,  # 0.15/0.85 to get 15% of original
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
```

---

## Phase 4: BERT Tokenization

### Step 4.1: Understanding BERT Tokenization

BERT doesn't work with raw text. It uses a tokenizer that:
1. Splits text into subword tokens (handles unknown words better)
2. Adds special tokens: [CLS] at start, [SEP] at end
3. Converts tokens to numerical IDs
4. Creates attention masks (tells BERT which tokens to focus on)

Example: "Hello, how are you?" becomes:
```
Tokens: [CLS] hello , how are you ? [SEP]
IDs: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1]
```

### Step 4.2: Initialize BERT Tokenizer

We'll use 'bert-base-uncased', a pre-trained BERT model:

```python
from transformers import BertTokenizer

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Tokenizer loaded successfully!")
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Test tokenization with a sample message
sample_text = "Free entry in 2 a wkly comp to win FA Cup final tkts"
print(f"\nSample text: {sample_text}")

# Tokenize
tokens = tokenizer.tokenize(sample_text)
print(f"\nTokens: {tokens}")

# Convert to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"\nToken IDs: {token_ids}")

# Full encoding (what BERT actually uses)
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
```

### Step 4.3: Analyze Message Lengths for Max Length Selection

BERT has a maximum sequence length (usually 512 tokens). We need to find an appropriate max_length:

```python
# Tokenize all messages to find length distribution
message_lengths = []

print("\nAnalyzing message lengths for max_length selection...")
for message in df['message_clean']:
    tokens = tokenizer.encode(message, add_special_tokens=True)
    message_lengths.append(len(tokens))

df['token_count'] = message_lengths

# Statistics
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
plt.show()

# Recommendation
max_length_95 = int(np.percentile(message_lengths, 95))
max_length_99 = int(np.percentile(message_lengths, 99))

print(f"\nRecommended max_length:")
print(f"  - 95% of messages fit in {max_length_95} tokens")
print(f"  - 99% of messages fit in {max_length_99} tokens")
print(f"  - Suggested: 128 tokens (covers most messages efficiently)")

# Count messages that will be truncated
MAX_LENGTH = 128
truncated = (df['token_count'] > MAX_LENGTH).sum()
print(f"\nWith max_length={MAX_LENGTH}:")
print(f"  - {truncated} messages ({truncated/len(df)*100:.2f}%) will be truncated")
print(f"  - {len(df)-truncated} messages ({(len(df)-truncated)/len(df)*100:.2f}%) fit completely")
```

### Step 4.4: Create PyTorch Dataset

BERT needs data in a specific format. We'll create a custom PyTorch Dataset:

```python
from torch.utils.data import Dataset

class SMSDataset(Dataset):
    """
    Custom Dataset for SMS messages
    Handles tokenization and prepares data for BERT
    """
    
    def __init__(self, messages, labels, tokenizer, max_length=128):
        """
        Args:
            messages: List of SMS text messages
            labels: List of labels (0 for ham, 1 for spam)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized example
        """
        message = str(self.messages[idx])
        label = self.labels[idx]
        
        # Tokenize the message
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
MAX_LENGTH = 128

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

# Verify dataset
print("\nSample from training dataset:")
sample = train_dataset[0]
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Attention Mask shape: {sample['attention_mask'].shape}")
print(f"Label: {sample['labels']}")
```

---

## Phase 5: Building and Configuring BERT Model

### Step 5.1: Load Pre-trained BERT Model

We'll use BertForSequenceClassification, which adds a classification head on top of BERT:

```python
from transformers import BertForSequenceClassification

# Load pre-trained BERT model with 2 output labels (ham and spam)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Move model to GPU if available
model = model.to(device)

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (float32)")
```

### Step 5.2: Understanding BERT Model Components

Let's understand what we just loaded:

```python
print("\n" + "="*60)
print("BERT COMPONENTS EXPLANATION")
print("="*60)

print("""
1. BERT Base Layer:
   - 12 transformer layers (attention + feedforward)
   - 768 hidden dimensions
   - 12 attention heads
   - Pre-trained on massive text corpus
   
2. Classifier Head:
   - Dense layer: 768 → 2 (ham or spam)
   - Dropout for regularization
   
3. How it works:
   a) Input text → Tokenizer → Token IDs
   b) Token IDs → BERT embeddings → Contextualized representations
   c) [CLS] token representation → Classifier → Prediction
   
The [CLS] token acts as an aggregate representation of the entire message.
""")
```

### Step 5.3: Configure Training Arguments

These settings control how the model learns:

```python
from transformers import TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model checkpoints
    num_train_epochs=3,               # Number of training epochs
    per_device_train_batch_size=16,   # Batch size for training
    per_device_eval_batch_size=32,    # Batch size for evaluation
    warmup_steps=500,                 # Warmup steps for learning rate
    weight_decay=0.01,                # Weight decay for regularization
    logging_dir='./logs',             # Directory for logs
    logging_steps=50,                 # Log every 50 steps
    evaluation_strategy='epoch',      # Evaluate at end of each epoch
    save_strategy='epoch',            # Save model at end of each epoch
    load_best_model_at_end=True,     # Load best model at end
    metric_for_best_model='f1',      # Use F1 score to determine best model
    greater_is_better=True,           # Higher F1 is better
    save_total_limit=2,               # Keep only 2 best checkpoints
    fp16=torch.cuda.is_available(),   # Use mixed precision if GPU available
    report_to='none',                 # Don't report to external services
    seed=42                           # For reproducibility
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

# Calculate total training steps
total_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
print(f"\nTotal training steps: {total_steps}")
print(f"Evaluation frequency: Every {training_args.logging_steps} steps")
```

### Step 5.4: Define Evaluation Metrics

We need to tell the model how to measure success:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(pred):
    """
    Compute evaluation metrics during training
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
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
   - (TP + TN) / (TP + TN + FP + FN)

2. Precision: Of predicted spam, how much is actually spam?
   - TP / (TP + FP)
   - Important: Minimize false positives (marking ham as spam)

3. Recall: Of actual spam, how much did we catch?
   - TP / (TP + FN)
   - Important: Catch as much spam as possible

4. F1-Score: Harmonic mean of precision and recall
   - 2 × (Precision × Recall) / (Precision + Recall)
   - Balanced measure for imbalanced datasets

Where:
- TP = True Positives (correctly identified spam)
- TN = True Negatives (correctly identified ham)
- FP = False Positives (ham marked as spam)
- FN = False Negatives (spam marked as ham)
""")
```

---

## Phase 6: Training the BERT Model

### Step 6.1: Initialize Trainer

The Trainer class handles the entire training loop:

```python
from transformers import Trainer

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n" + "="*60)
print("TRAINER INITIALIZED")
print("="*60)
print("Ready to start training!")
```

### Step 6.2: Train the Model

This is where the magic happens! Training can take 10-30 minutes on GPU, longer on CPU:

```python
import time

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print("This may take several minutes...")
print("You'll see progress bars and periodic evaluation results\n")

# Record start time
start_time = time.time()

# Train the model
train_result = trainer.train()

# Record end time
end_time = time.time()
training_time = end_time - start_time

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Total training time: {training_time/60:.2f} minutes")
print(f"Average time per epoch: {training_time/training_args.num_train_epochs/60:.2f} minutes")

# Save training metrics
train_metrics = train_result.metrics
print("\nFinal Training Metrics:")
for key, value in train_metrics.items():
    print(f"  {key}: {value:.4f}")
```

### Step 6.3: Visualize Training Progress

Plot how the model improved over time:

```python
import json

# Load training logs
with open('./results/trainer_state.json', 'r') as f:
    trainer_state = json.load(f)

log_history = trainer_state['log_history']

# Extract metrics
train_loss = []
eval_loss = []
eval_accuracy = []
eval_f1 = []
steps = []

for entry in log_history:
    if 'loss' in entry:  # Training logs
        train_loss.append(entry['loss'])
        steps.append(entry['step'])
    if 'eval_loss' in entry:  # Evaluation logs
        eval_loss.append(entry['eval_loss'])
        eval_accuracy.append(entry['eval_accuracy'])
        eval_f1.append(entry['eval_f1'])

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training Loss
axes[0, 0].plot(steps, train_loss, marker='o', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].grid(alpha=0.3)

# Validation Loss
epochs = list(range(1, len(eval_loss) + 1))
axes[0, 1].plot(epochs, eval_loss, marker='s', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss per Epoch')
axes[0, 1].grid(alpha=0.3)

# Validation Accuracy
axes[1, 0].plot(epochs, eval_accuracy, marker='s', linewidth=2, markersize=8, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Validation Accuracy per Epoch')
axes[1, 0].set_ylim([0.9, 1.0])
axes[1, 0].grid(alpha=0.3)

# Validation F1 Score
axes[1, 1].plot(epochs, eval_f1, marker='s', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('Validation F1 Score per Epoch')
axes[1, 1].set_ylim([0.9, 1.0])
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=300)
plt.show()

print("\nTraining progress visualization saved!")
```

---

## Phase 7: Model Evaluation

### Step 7.1: Evaluate on Test Set

Now let's see how well the model performs on completely unseen data:

```python
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)

print("\nTest Set Results:")
for key, value in test_results.items():
    if key.startswith('eval_'):
        metric_name = key.replace('eval_', '').upper()
        print(f"  {metric_name}: {value:.4f}")
```

### Step 7.2: Generate Predictions and Detailed Analysis

Get predictions for detailed analysis:

```python
# Get predictions
predictions_output = trainer.predict(test_dataset)
predictions = predictions_output.predictions.argmax(-1)
true_labels = predictions_output.label_ids

# Calculate comprehensive metrics
from