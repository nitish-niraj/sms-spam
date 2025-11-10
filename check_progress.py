"""Check training progress"""
import os
from datetime import datetime

print("="*60)
print("TRAINING PROGRESS CHECK")
print("="*60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check log file
if os.path.exists('training_output.log'):
    with open('training_output.log', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"Log file lines: {len(lines)}")
    
    # Get last 20 lines
    print("\nLast 20 lines of output:")
    print("-" * 60)
    for line in lines[-20:]:
        print(line.rstrip())
else:
    print("⚠️  Log file not found yet...")

# Check for model directory
if os.path.exists('saved_model'):
    print("\n✅ Model directory exists - Training completed!")
else:
    print("\n⏳ Model not saved yet - Training in progress...")

# Check for visualization files
print("\nGenerated files:")
for f in ['class_distribution.png', 'message_length_analysis.png', 
          'token_length_analysis.png', 'training_progress.png', 
          'confusion_matrix.png', 'model_summary.json']:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✅ {f} ({size:.1f} KB)")
    else:
        print(f"  ⏳ {f} - not yet created")

print("\n" + "="*60)
