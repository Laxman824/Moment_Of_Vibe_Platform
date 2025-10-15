# Demo Video Script (5-10 minutes)

## Introduction (1 minute)
- Hello, I'm presenting the Emotion Analytics Module for Moment of Vibe
- This module detects emotions from voice calls in real-time
- Built with OpenSmile features and PyTorch neural networks

## Quick Overview (1 minute)
- Show project structure
- Highlight key components: features.py, model.py, pipeline.py
- Mention test coverage: >80%

## Data Preparation (1 minute)
- Show RAVDESS dataset structure
- Run: `python scripts/prepare_data.py`
- Explain feature extraction with OpenSmile

## Training Demo (2 minutes)
- Run: `python scripts/train.py --epochs 10`  # Abbreviated for demo
- Show training progress
- Explain accuracy metrics

## Inference Demo (2-3 minutes)
- Load a sample audio file
- Run pipeline.process_audio()
- Show output:
  - Emotion scores
  - Audio quality metrics
  - Context-aware suggestions
- Demonstrate real-time speed

## Evaluation Results (2 minutes)
- Run: `python scripts/evaluate.py`
- Show metrics: >75% accuracy ✅
- Display confusion matrices and ROC curves
- Performance: <1.5s processing time ✅

## Integration & Future Work (1 minute)
- Show ProcessingResult interface
- Explain MOV integration path
- Mention ethical considerations
- Future Enhancements
- Integration with Whisper for transcript-based emotion refinement
-  Claude Sonnet 4.5 integration for advanced nudges
- Multilingual emotion detection
- Real-time streaming support (currently chunk-based)
- Active learning pipeline for continuous improvement
- Confidence calibration for better probability estimates
- Thank you!

## Commands for Demo

```bash
# Setup
pip install -r requirements.txt

# Data preparation
python scripts/prepare_data.py

# Training
python scripts/train.py --epochs 20

# Evaluation
python scripts/evaluate.py --benchmark

# Interactive demo
python -c "
from src.pipeline import EmotionAnalyticsPipeline
pipeline = EmotionAnalyticsPipeline('models/checkpoints/best_model.pth')
result = pipeline.process_audio('data/raw/sample.wav')
print(result)
"