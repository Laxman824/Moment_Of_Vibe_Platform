
#!/bin/bash

echo "=== DEMO: Emotion Analytics Module ==="

# 1. Show project structure
echo -e "\n1. Project Structure:"
tree -L 2 -I '__pycache__|*.pyc|.git|htmlcov|.pytest_cache|*.egg-info'

# 2. Show test coverage
echo -e "\n2. Test Coverage (>80% required):"
python3 -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -q | tail -n 20
# 3. Process sample audio
echo -e "\n3. Processing Sample Audio:"
python3 << 'PYTHON'
from src.pipeline import EmotionAnalyticsPipeline
from src.config import RAW_DATA_DIR, MODELS_DIR
from pathlib import Path
import json
import time
import sys

# Find a sample WAV file
raw_files = list(RAW_DATA_DIR.rglob("*.wav"))

if not raw_files:
    print("❌ No WAV files found in data/raw directory")
    print(f"   Please add WAV files to: {RAW_DATA_DIR}")
    sys.exit(1)

# Use the first available WAV file
sample_file = raw_files[0]
print(f"📁 Using sample file: {sample_file.name}")
print(f"   Location: {sample_file.parent}")
print("-" * 60)

# Check if model exists
model_path = MODELS_DIR / "emotion_model.pth"
if not model_path.exists():
    print(f"❌ Model not found at: {model_path}")
    print("   Run training first: python3 scripts/train_model.py")
    sys.exit(1)

# Initialize pipeline with model
try:
    print(f"🔧 Loading model from: {model_path}")
    pipeline = EmotionAnalyticsPipeline(model_path=str(model_path))
    print("✅ Pipeline initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 60)

# Process audio
start_time = time.time()
try:
    result = pipeline.process_audio(str(sample_file))
    duration = time.time() - start_time
    
    # Extract emotions from result
    emotions = result.get('emotions', {})
    suggestions = result.get('suggestions', [])
    audio_quality = result.get('audio_quality', {})
    metadata = result.get('metadata', {})
    
    # Pretty print results
    print("\n🎯 PREDICTION RESULTS:")
    print("=" * 60)
    
    # Get dominant emotion
    if emotions:
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        print(f"  🎭 Dominant Emotion: {dominant_emotion[0].upper()}")
        print(f"  📊 Confidence:       {dominant_emotion[1]:.2%}")
        
        # Calculate valence and arousal from emotions
        valence = emotions.get('joy', 0) - emotions.get('anger', 0)
        arousal = (emotions.get('energy', 0) + emotions.get('anger', 0)) / 2
        
        print(f"  😊 Valence:          {valence:.3f} (range: -1 to 1)")
        print(f"  ⚡ Arousal:          {arousal:.3f} (range: 0 to 1)")
        
        print("\n  📈 All Emotion Scores:")
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"     {emotion.capitalize():12s} {bar} {score:.2%}")
    else:
        print("  ⚠️  No emotion predictions available")
    
    print(f"\n  ⏱️  Processing Time: {duration:.3f}s {'✅' if duration < 1.5 else '⚠️'} (Required: <1.5s)")
    
    # Audio quality
    if audio_quality:
        print(f"\n  🔊 Audio Quality:")
        print(f"     Quality Level:   {audio_quality.get('quality_level', 'Unknown')}")
        print(f"     SNR:            {audio_quality.get('snr_db', 0):.1f} dB")
        print(f"     Quality Score:   {audio_quality.get('quality_score', 0):.2%}")
    
    # Suggestions
    if suggestions:
        print(f"\n  💡 Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"     {i}. {suggestion}")
    
    # Metadata
    if metadata:
        print(f"\n  ℹ️  Metadata:")
        print(f"     Audio Duration:  {metadata.get('audio_duration_seconds', 0):.2f}s")
        print(f"     Sample Rate:     {metadata.get('sample_rate', 0)} Hz")
        print(f"     Features:        {metadata.get('feature_count', 0)}")
    
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error processing audio: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON

# 4. Show model performance
echo -e "\n4. Model Performance Metrics:"
python3 << 'PYTHON'
from pathlib import Path
import json

print("=" * 60)

# Try to load actual metrics if saved
metrics_file = Path("models/metrics.json")
checkpoint_metrics = Path("models/checkpoints/evaluation/metrics.json")

# Try checkpoint metrics first (usually more detailed)
if checkpoint_metrics.exists():
    metrics_path = checkpoint_metrics
elif metrics_file.exists():
    metrics_path = metrics_file
else:
    metrics_path = None

if metrics_path:
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"📊 Metrics from: {metrics_path.name}")
        print("-" * 60)
        
        # Debug: print all keys
        print(f"Available keys: {list(metrics.keys())}")
        print()
        
        # Try different possible keys for accuracy
        acc = (metrics.get('val_accuracy') or 
               metrics.get('accuracy') or
               metrics.get('test_accuracy') or 
               metrics.get('weighted avg', {}).get('precision', 0))
        
        # Try different possible keys for F1 score  
        f1 = (metrics.get('val_f1') or
              metrics.get('f1_score') or
              metrics.get('test_f1') or
              metrics.get('weighted avg', {}).get('f1-score', 0))
        
        # Display overall metrics
        if acc:
            print(f"  🎯 Accuracy:  {acc*100 if acc <= 1 else acc:.2f}%")
        if f1:
            print(f"  📈 F1 Score:  {f1:.3f}")
        
        # Try to show class-specific metrics
        for key in metrics.keys():
            if key in ['angry', 'happy', 'neutral', 'sad', 'fear', 'disgust', 'surprise']:
                print(f"\n  📊 {key.capitalize()}:")
                if isinstance(metrics[key], dict):
                    for metric, value in metrics[key].items():
                        print(f"     {metric}: {value:.3f}")
        
        # Show macro/weighted averages
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in metrics:
                print(f"\n  📊 {avg_type.title()}:")
                for metric, value in metrics[avg_type].items():
                    if isinstance(value, (int, float)):
                        print(f"     {metric}: {value:.3f}")
        
        # Show additional info
        if 'best_epoch' in metrics:
            print(f"\n  🏆 Best Epoch: {metrics['best_epoch']}")
        if 'best_val_loss' in metrics:
            print(f"  📉 Best Val Loss: {metrics['best_val_loss']:.4f}")
            
    except Exception as e:
        print(f"❌ Error reading metrics: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  No metrics found. Train model first:")
    print("   python3 scripts/train_model.py")

print("=" * 60)
PYTHON

# 5. Show available models
echo -e "\n5. Available Models:"
python3 << 'PYTHON'
from pathlib import Path
import os

print("=" * 60)
print("📦 Model Files:")
print("-" * 60)

model_dir = Path("models")
model_files = []

# Find all model-related files
for ext in ['*.pth', '*.pt', '*.pkl', '*.json']:
    model_files.extend(model_dir.rglob(ext))

if model_files:
    for file in sorted(model_files):
        size = os.path.getsize(file)
        size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        rel_path = file.relative_to(model_dir)
        
        # Add emoji based on file type
        if file.suffix == '.pth' or file.suffix == '.pt':
            emoji = "🧠"
        elif file.suffix == '.pkl':
            emoji = "📊"
        elif file.suffix == '.json':
            emoji = "📋"
        else:
            emoji = "📄"
        
        print(f"  {emoji} {str(rel_path):<40} {size_str:>10}")
else:
    print("  ⚠️  No model files found. Run training first.")

print("=" * 60)
PYTHON

echo -e "\n=== ✅ DEMO COMPLETE ==="
echo ""
echo "Next steps:"
echo "  • View detailed coverage: open htmlcov/index.html"
echo "  • Run training: python3 scripts/train_model.py"
echo "  • Run evaluation: python3 scripts/evaluate_model.py"
echo "  • Process custom audio: python3 -c \"from src.pipeline import EmotionAnalyticsPipeline; pipeline = EmotionAnalyticsPipeline(model_path='models/emotion_model.pth'); print(pipeline.process_audio('your_file.wav'))\""
echo ""