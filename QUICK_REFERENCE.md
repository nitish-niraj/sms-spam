# SMS Spam Detection - Quick Reference

## 🚀 Quick Start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (15-30 min on GPU)
python sms_spam_bert.py

# 3. Test predictions
python demo.py
```

## 📝 Common Commands

### Training
```bash
# Full training pipeline
python sms_spam_bert.py

# Validate setup (no training)
python test_implementation.py
```

### Predictions
```bash
# Interactive demo
python demo.py

# Single prediction
python predict.py "Your message here"

# Example
python predict.py "WINNER! Claim your prize now!"
```

### Python API
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Predict
def predict(text):
    enc = tokenizer.encode_plus(text, max_length=128, padding='max_length',
                                 truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(enc['input_ids'].to(device), 
                   enc['attention_mask'].to(device))
        probs = torch.softmax(out.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return 'spam' if pred == 1 else 'ham', probs[0][pred].item()

# Use
label, conf = predict("FREE prize!")
print(f"{label} ({conf:.2%})")
```

## 📊 Key Files

| File | Purpose | Size |
|------|---------|------|
| sms_spam_bert.py | Main training script | 25KB |
| demo.py | Interactive demo | 4KB |
| predict.py | CLI predictions | 3KB |
| README.md | Full documentation | 9KB |
| USER_GUIDE.md | Detailed guide | 8KB |

## 🎯 Expected Performance

| Metric | Target |
|--------|--------|
| Accuracy | >98% |
| Precision | >95% |
| Recall | >90% |
| F1-Score | >95% |

## 📦 Generated Files

After training, these files are created:
- `class_distribution.png` - Class balance chart
- `message_length_analysis.png` - Length stats
- `token_length_analysis.png` - Token distribution
- `training_progress.png` - Training curves
- `confusion_matrix.png` - Performance matrix
- `model_summary.json` - Metrics summary
- `saved_model/` - Trained model directory

## ⚙️ Configuration

Edit `sms_spam_bert.py` to change:

```python
# Training parameters
training_args = TrainingArguments(
    num_train_epochs=3,              # Number of epochs
    per_device_train_batch_size=16,  # Batch size
    learning_rate=5e-5,              # Learning rate
    warmup_steps=500,                # Warmup steps
    weight_decay=0.01,               # Regularization
)

# Sequence length
MAX_LENGTH = 128  # Token limit
```

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size
per_device_train_batch_size=8  # or 4
```

### "Model not found"
```bash
# Train first
python sms_spam_bert.py
```

### "Slow on CPU"
```bash
# Use fewer epochs
num_train_epochs=1
```

### "Module not found"
```bash
# Install all deps
pip install -r requirements.txt
```

## ⏱️ Training Time

| Hardware | Time (3 epochs) |
|----------|----------------|
| GPU (T4) | ~15-20 min |
| GPU (RTX 3090) | ~8-10 min |
| CPU (4 cores) | ~2-3 hours |

## 💾 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning |
| transformers | ≥4.30.0 | BERT model |
| pandas | ≥1.5.0 | Data handling |
| scikit-learn | ≥1.2.0 | Metrics |
| matplotlib | ≥3.6.0 | Plotting |

## 🎓 Model Details

| Property | Value |
|----------|-------|
| Architecture | BERT base uncased |
| Parameters | ~110 million |
| Vocabulary | 30,522 tokens |
| Max Length | 128 tokens |
| Layers | 12 transformer layers |
| Hidden Size | 768 dimensions |
| Attention Heads | 12 heads |

## 📚 Dataset

| Property | Value |
|----------|-------|
| Total Messages | 5,572 |
| Spam | 747 (13.4%) |
| Ham | 4,825 (86.6%) |
| Train | 3,900 (70%) |
| Validation | 836 (15%) |
| Test | 836 (15%) |

## 🔗 Resources

- [Full README](README.md)
- [User Guide](USER_GUIDE.md)
- [Changelog](CHANGELOG.md)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Transformers Docs](https://huggingface.co/docs/transformers/)

## 💡 Tips

1. **Use GPU** for 20x speedup
2. **Monitor F1 score** not just accuracy
3. **Check false positives** - blocking ham is bad
4. **Save checkpoints** during training
5. **Retrain periodically** as spam evolves
6. **Use validation set** to tune hyperparameters
7. **Set confidence thresholds** based on your needs

## 🎯 Prediction Examples

```python
# Spam examples
predict("FREE entry to win cash!")          # → spam
predict("WINNER! Claim your prize")         # → spam
predict("Call 12345 for exclusive offer")   # → spam

# Ham examples
predict("Hey, how are you?")                # → ham
predict("Meeting at 3pm tomorrow")          # → ham
predict("Thanks for the help!")             # → ham
```

---

**Quick Help**: See [README.md](README.md) for detailed instructions.
