# SMS Spam Detection with BERT - User Guide

## Quick Start

This guide will walk you through using the SMS Spam Detection system.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For CPU-only (faster installation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Option 1: Complete Training Pipeline

To train the model from scratch and generate all visualizations:

```bash
python sms_spam_bert.py
```

This will:
- Load and analyze the SMS dataset
- Train a BERT model (3 epochs, ~15-30 min on GPU, 2-3 hours on CPU)
- Generate visualizations and performance metrics
- Save the trained model to `./saved_model/`

**Output files:**
- `class_distribution.png` - Dataset class balance
- `message_length_analysis.png` - Message length statistics
- `token_length_analysis.png` - Token distribution
- `training_progress.png` - Training metrics over time
- `confusion_matrix.png` - Model performance visualization
- `model_summary.json` - Performance metrics
- `saved_model/` - Trained model directory

### Option 2: Interactive Demo

If you already have a trained model, use the demo script:

```bash
python demo.py
```

This provides:
- Pre-defined test cases
- Interactive mode to test your own messages
- Real-time predictions with confidence scores

### Option 3: Python API

Use the model programmatically:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def predict_spam(text, max_length=128):
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
    
    return ('spam' if pred == 1 else 'ham', conf)

# Example
message = "WINNER! Claim your prize now!"
label, confidence = predict_spam(message)
print(f"{label} ({confidence:.2%} confidence)")
```

## Understanding the Results

### Model Performance Metrics

- **Accuracy**: Overall correctness (expected: >98%)
- **Precision**: Of messages flagged as spam, how many are actually spam? (expected: >95%)
- **Recall**: Of actual spam messages, how many did we catch? (expected: >90%)
- **F1-Score**: Balanced measure combining precision and recall (expected: >95%)

### Confusion Matrix

The confusion matrix shows:
- True Negatives (TN): Ham correctly identified as ham
- True Positives (TP): Spam correctly identified as spam
- False Positives (FP): Ham incorrectly marked as spam (minimize this!)
- False Negatives (FN): Spam that slipped through (catch these!)

### Visualizations Explained

1. **Class Distribution**: Shows dataset balance (ham vs spam)
2. **Message Length Analysis**: Character/word count distributions
3. **Token Length Analysis**: BERT token counts and percentiles
4. **Training Progress**: Loss, accuracy, and F1 score over epochs
5. **Confusion Matrix**: Detailed performance breakdown

## Configuration

### Adjusting Training Parameters

Edit `sms_spam_bert.py` and modify the `TrainingArguments`:

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # More epochs = better learning (but slower)
    per_device_train_batch_size=16,   # Smaller = less memory, slower training
    per_device_eval_batch_size=32,    # Can be larger than train batch
    warmup_steps=500,                 # Learning rate warmup
    weight_decay=0.01,                # Regularization strength
    learning_rate=5e-5,               # Default BERT learning rate
)
```

### Memory Issues?

If you encounter out-of-memory errors:

1. **Reduce batch size:**
   ```python
   per_device_train_batch_size=8  # or even 4
   ```

2. **Reduce max sequence length:**
   ```python
   MAX_LENGTH = 64  # instead of 128
   ```

3. **Use CPU instead of GPU:**
   ```python
   device = torch.device('cpu')
   ```

### Speed up Training

1. **Use GPU** (20x faster than CPU)
2. **Enable mixed precision:**
   ```python
   fp16=True  # in TrainingArguments
   ```
3. **Reduce epochs for testing:**
   ```python
   num_train_epochs=1  # quick test
   ```

## Common Use Cases

### Case 1: Quick Evaluation

Just want to see results without full training:
```bash
# Reduce to 1 epoch for quick test
# Edit sms_spam_bert.py: num_train_epochs=1
python sms_spam_bert.py
```

### Case 2: Production Deployment

For production use:
1. Train with full 3+ epochs
2. Evaluate on holdout test set
3. Monitor false positive rate
4. Implement confidence thresholding
5. Set up periodic retraining

### Case 3: Custom Dataset

To use your own dataset:

1. Format as tab-separated file: `label\tmessage`
2. Update dataset path in script
3. Ensure labels are 'ham' and 'spam'
4. Run training pipeline

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU

### Issue: "Model not found"
**Solution**: Run `python sms_spam_bert.py` first to train the model

### Issue: "Slow training on CPU"
**Solution**: Use Google Colab with free GPU or reduce epochs

### Issue: "Connection timeout"
**Solution**: BERT model downloads on first run. Check internet connection.

### Issue: "ImportError"
**Solution**: Install all dependencies: `pip install -r requirements.txt`

## Performance Benchmarks

### Training Time
- **GPU (T4/V100)**: ~15-20 minutes (3 epochs)
- **CPU (4 cores)**: ~2-3 hours (3 epochs)
- **GPU (RTX 3090)**: ~8-10 minutes (3 epochs)

### Model Size
- **On disk**: ~440 MB
- **In memory**: ~440 MB (float32)
- **In memory**: ~220 MB (float16 with mixed precision)

### Inference Speed
- **GPU**: ~100-200 messages/second
- **CPU**: ~10-20 messages/second

## Best Practices

1. **Always use stratified split** to maintain class balance
2. **Monitor both precision and recall** for spam detection
3. **Analyze false positives** - blocking legitimate messages is bad!
4. **Set confidence thresholds** based on your tolerance for errors
5. **Retrain periodically** as spam patterns evolve
6. **Use validation set** to prevent overfitting
7. **Save best model** based on F1 score, not just accuracy

## Advanced Topics

### Transfer Learning

The BERT model is already pre-trained. We're fine-tuning it for spam detection.
The pre-training gives it language understanding; fine-tuning specializes it.

### Attention Mechanisms

BERT uses attention to focus on important words. For example, in spam:
- "FREE" gets high attention
- "WINNER" gets high attention
- "you" gets low attention

### Tokenization

BERT uses WordPiece tokenization:
- Common words: single token ("free" → "free")
- Rare words: multiple tokens ("cryptography" → "cry", "##pto", "##graphy")
- Handles unknown words gracefully

## Contributing

Improvements welcome:
- Better visualization
- Web interface
- Multi-language support
- Ensemble models
- Real-time API

## Support

For issues or questions:
1. Check this guide
2. Review error messages
3. Open GitHub issue
4. Include error logs and system info

## Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SMS Spam Dataset Info](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

**Happy Spam Detection!** 🎯🚫📱
