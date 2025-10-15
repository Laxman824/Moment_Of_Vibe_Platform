"""
Configuration module for Emotion Analytics.
Centralizes all hyperparameters, paths, and constants.
"""
from pathlib import Path
from typing import Dict, List
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Emotion labels (aligned with MOV requirements)
EMOTION_LABELS: List[str] = ["anger", "joy", "energy", "confidence"]
NUM_EMOTIONS = len(EMOTION_LABELS)

# Audio processing parameters
SAMPLE_RATE = 16000  # Standard for speech processing
CHUNK_SIZE_SECONDS = 10  # Process audio in 10-second chunks
CHUNK_SIZE_SAMPLES = CHUNK_SIZE_SECONDS * SAMPLE_RATE
MIN_CHUNK_DURATION = 1.0  # Minimum chunk duration in seconds

# OpenSmile feature extraction
OPENSMILE_FEATURE_SET = "eGeMAPSv02"  # Extended Geneva Minimalistic Acoustic Parameter Set
OPENSMILE_FEATURE_LEVEL = "Functionals"  # Extract statistical functionals

# Audio quality thresholds
SNR_THRESHOLD = 20.0  # dB - Signal-to-noise ratio threshold
MIN_RMS_ENERGY = 0.01  # Minimum RMS energy for non-silent audio
MAX_CLIPPING_RATIO = 0.01  # Maximum acceptable clipping ratio

# Model architecture
MODEL_CONFIG: Dict = {
    "input_size": 88,  # eGeMAPSv02 has 88 features
    "hidden_sizes": [128, 64],
    "output_size": NUM_EMOTIONS,
    "dropout": 0.3,
    "activation": "relu",
}

# Training parameters
TRAINING_CONFIG: Dict = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "early_stopping_patience": 10,
    "validation_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
}

# Performance requirements (from assignment)
ACCURACY_THRESHOLD = 0.75  # 75% minimum accuracy
PROCESSING_TIME_THRESHOLD = 1.5  # seconds per 10-second chunk

# ProcessingResult interface structure (matching MOV docs)
PROCESSING_RESULT_SCHEMA = {
    "transcript": str,  # Mock empty for this module
    "emotions": Dict[str, float],  # Emotion scores 0-1
    "audio_quality": Dict[str, float],  # Quality metrics
    "suggestions": List[str],  # Context-aware suggestions
    "metadata": Dict,  # Additional info
}

# Suggestion templates (rule-based)
SUGGESTION_RULES = {
    "anger": {
        "threshold": 0.5,
        "high": "Consider taking a calming breath. The conversation seems tense.",
        "medium": "The energy is rising. Would a topic shift help?",
    },
    "joy": {
        "threshold": 0.7,
        "high": "Great vibes! This seems like a positive moment.",
    },
    "energy": {
        "threshold": 0.3,
        "low": "Energy seems low. Maybe suggest a topic that excites both parties?",
    },
    "confidence": {
        "threshold": 0.4,
        "low": "Encourage your conversation partner with active listening.",
    },
}

# Device configuration
DEVICE = "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Bias mitigation notes (for documentation)
BIAS_MITIGATION_NOTES = """
- Dataset should include diverse accents (American, British, Indian, etc.)
- Balance gender representation in training data
- Test on underrepresented groups (non-native speakers)
- Document performance disparities across demographics
- Consider cultural differences in emotional expression
"""