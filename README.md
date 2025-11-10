# SMS Spam Detection with BERT

A state-of-the-art SMS spam detection system using BERT (Bidirectional Encoder Representations from Transformers), a powerful deep learning model for natural language processing.

## Project Overview

This project implements a complete end-to-end spam detection system that:
- Uses pre-trained BERT model for text classification
- Achieves high accuracy in detecting spam messages
- Includes comprehensive data analysis and visualization
- Provides an easy-to-use prediction interface

## Dataset

The project uses the SMS Spam Collection dataset containing 5,574 SMS messages labeled as spam or ham (legitimate). The dataset is included in the `sms+spam+collection/` directory.

## Features

- **Advanced NLP**: Utilizes BERT's bidirectional context understanding
- **Comprehensive Analysis**: Detailed exploratory data analysis with visualizations
- **High Performance**: Achieves >95% accuracy on test data
- **Easy to Use**: Simple prediction function for new messages
- **GPU Support**: Automatic GPU detection and utilization if available

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading pre-trained BERT models)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nitish-niraj/sms-spam.git
cd sms-spam
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

**Note**: On first run, the script will download the pre-trained BERT model (~440MB) from Hugging Face. This requires internet access and may take a few minutes.

### Optional: GPU Support

For faster training with NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Training the Model

Run the main script to train the model and generate all visualizations:

```bash
python sms_spam_bert.py
```

The script will:
1. Load and analyze the dataset
2. Preprocess the data
3. Train the BERT model (takes 10-30 minutes on GPU, longer on CPU)
4. Evaluate on test set
5. Save the trained model and generate visualizations

### Generated Files

After running the script, you'll have:

- **Visualizations**:
  - `class_distribution.png` - Distribution of spam vs ham messages
  - `message_length_analysis.png` - Analysis of message lengths
  - `token_length_analysis.png` - Token distribution analysis
  - `training_progress.png` - Training metrics over time
  - `confusion_matrix.png` - Model performance visualization

- **Model Files**:
  - `saved_model/` - Directory containing trained model and tokenizer
  - `model_summary.json` - Summary of model performance

- **Training Logs**:
  - `results/` - Training checkpoints and logs
  - `logs/` - TensorBoard logs

### Using the Trained Model

#### Option 1: Command Line Interface

For quick single predictions:

```bash
python predict.py "WINNER! You have won a $1000 prize. Call now!"
```

#### Option 2: Interactive Demo

Run the interactive demo for testing multiple messages:

```bash
python demo.py
```

#### Option 3: Python API

To use the trained model programmatically:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prediction function
def predict_spam(text, model, tokenizer, device, max_length=128):
    model.eval()
    
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
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = 'spam' if prediction == 1 else 'ham'
    return label, confidence

# Example usage
message = "WINNER! You have won a $1000 prize. Call now!"
label, confidence = predict_spam(message, model, tokenizer, device)
print(f"Prediction: {label} (confidence: {confidence:.4f})")
```

## Project Structure

```
sms-spam/
├── sms+spam+collection/
│   ├── SMSSpamCollection    # Dataset file (5,572 SMS messages)
│   └── readme               # Dataset information
├── sms_spam_bert.py          # Main implementation script (complete pipeline)
├── demo.py                   # Interactive demo for testing predictions
├── predict.py                # Command-line prediction tool
├── test_implementation.py    # Validation script for testing setup
├── sms_spam_bert_roadmap.md  # Detailed project roadmap and tutorial
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── USER_GUIDE.md             # Comprehensive user guide
└── saved_model/              # Trained model (generated after training)
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## Available Scripts

1. **sms_spam_bert.py** - Complete training pipeline
   - Loads and analyzes dataset
   - Trains BERT model
   - Generates all visualizations
   - Saves trained model
   
2. **demo.py** - Interactive demo
   - Test with pre-defined messages
   - Interactive mode for custom messages
   - Real-time predictions
   
3. **predict.py** - Command-line tool
   - Single-line predictions
   - Batch processing friendly
   - JSON output option
   
4. **test_implementation.py** - Setup validation
   - Tests data loading
   - Verifies tokenizer
   - Checks dependencies

## Model Architecture

The project uses `bert-base-uncased` with:
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- ~110 million parameters
- Fine-tuned classification head for binary classification

## Performance

Expected results on test set:
- **Accuracy**: >98%
- **Precision**: >95%
- **Recall**: >90%
- **F1-Score**: >95%

## Understanding BERT

BERT (Bidirectional Encoder Representations from Transformers) is revolutionary because:

1. **Pre-trained Knowledge**: Trained on massive text corpora (Wikipedia, books)
2. **Bidirectional Context**: Understands words in both directions
3. **Semantic Understanding**: Captures meaning and relationships
4. **Attention Mechanism**: Focuses on relevant parts of text

Unlike traditional methods (TF-IDF, word frequencies), BERT understands context and semantic meaning.

## Training Details

- **Epochs**: 3
- **Batch Size**: 16 (training), 32 (evaluation)
- **Learning Rate**: 5e-5 (default with warmup)
- **Optimizer**: AdamW
- **Max Sequence Length**: 128 tokens
- **Training Time**: ~15-20 minutes on GPU, ~2-3 hours on CPU

## Data Split

- **Training**: 70% (3,900 messages)
- **Validation**: 15% (836 messages)
- **Test**: 15% (836 messages)

## Requirements

- torch>=2.0.0
- transformers>=4.30.0
- pandas>=1.5.0
- numpy>=1.23.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- scikit-learn>=1.2.0
- tqdm>=4.65.0

## Troubleshooting

### Out of Memory Error

If you encounter GPU memory issues:
1. Reduce batch size in training_args
2. Use CPU instead (slower but more memory)
3. Reduce max_length to 64 tokens

### Slow Training

- Use GPU if available (20x faster than CPU)
- Consider using Google Colab with free GPU
- Reduce number of epochs for faster results

### Module Not Found

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Google Colab

To run on Google Colab with free GPU:

1. Upload the notebook to Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Upload the dataset or clone the repository
4. Run all cells

## Future Enhancements

Potential improvements:
- Web interface for real-time predictions
- Mobile app integration
- Multi-language support
- Additional models (RoBERTa, DistilBERT)
- Ensemble methods
- Active learning pipeline

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Acknowledgments

- SMS Spam Collection Dataset
- Hugging Face Transformers library
- BERT paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a educational/research project. For production use, consider additional validation, error handling, and monitoring.
