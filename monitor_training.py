"""Monitor training progress without interrupting the process"""
import os
import time
from datetime import datetime

def check_progress():
    print("\n" + "="*70)
    print(f"  TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check if Python process is running
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        python_running = 'python.exe' in result.stdout
        
        if python_running:
            print("✅ Training process is RUNNING")
        else:
            print("⚠️  No Python training process detected")
    except:
        print("❓ Could not check process status")
    
    # Check for generated files
    print("\n📁 Generated Files:")
    print("-" * 70)
    
    files_to_check = [
        ('class_distribution.png', 'Data exploration visualization'),
        ('message_length_analysis.png', 'Message length distribution'),
        ('token_length_analysis.png', 'Token analysis'),
        ('training_progress.png', 'Training metrics over time'),
        ('confusion_matrix.png', 'Model performance matrix'),
        ('model_summary.json', 'Training summary'),
        ('saved_model', 'Trained BERT model (directory)')
    ]
    
    completed = 0
    total = len(files_to_check)
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            if os.path.isdir(filename):
                print(f"  ✅ {filename:<30} - {description}")
            else:
                size = os.path.getsize(filename) / 1024
                print(f"  ✅ {filename:<30} - {description} ({size:.1f} KB)")
            completed += 1
        else:
            print(f"  ⏳ {filename:<30} - {description}")
    
    # Progress bar
    progress = (completed / total) * 100
    bar_length = 50
    filled = int(bar_length * completed / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\n📊 Overall Progress: {progress:.0f}%")
    print(f"[{bar}] {completed}/{total} files")
    
    # Training phases
    print("\n📋 Training Phases:")
    print("-" * 70)
    phases = [
        ("Phase 1-3: Data Loading & Preprocessing", completed >= 3),
        ("Phase 4: BERT Tokenization", completed >= 3),
        ("Phase 5-6: Model Training", completed >= 4),
        ("Phase 7: Evaluation", completed >= 5),
        ("Phase 8: Model Saving", completed >= 6)
    ]
    
    for phase, done in phases:
        status = "✅" if done else "⏳"
        print(f"  {status} {phase}")
    
    # Estimates
    if completed == total:
        print("\n🎉 TRAINING COMPLETED! 🎉")
        print("\nYou can now use the model with:")
        print("  - python demo.py (interactive demo)")
        print("  - python predict.py \"your message here\"")
    elif python_running:
        remaining_files = total - completed
        if completed >= 3:
            print(f"\n⏰ Estimated: Training in progress (Phase {completed-2}/6)")
            print("   This may take 2-3 hours on CPU")
        else:
            print("\n⏰ Initializing... (loading libraries and data)")
    else:
        print("\n⚠️  Training may have stopped. Check the training window.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    os.chdir('e:/SMS')
    check_progress()
