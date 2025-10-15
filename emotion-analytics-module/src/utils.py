"""
Utility functions for the Emotion Analytics module.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def load_audio(
    audio_path: Path | str,
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with error handling and optional resampling.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        normalize: Whether to normalize audio to [-1, 1]
    
    Returns:
        Tuple of (audio_data, sample_rate)
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format is invalid
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        audio, sr = sf.read(str(audio_path))
        
        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed (using simple linear interpolation)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        
        # Normalize
        if normalize and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        logger.debug(f"Loaded audio: {audio_path}, shape: {audio.shape}, sr: {sr}")
        return audio, sr
        
    except Exception as e:
        raise ValueError(f"Error loading audio file {audio_path}: {str(e)}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Simple audio resampling using linear interpolation.
    For production, use librosa.resample for better quality.
    
    Args:
        audio: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    
    # Simple linear interpolation
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)
    
    return resampled


def chunk_audio(
    audio: np.ndarray,
    chunk_size: int,
    overlap: float = 0.0
) -> list[np.ndarray]:
    """
    Split audio into chunks for processing.
    
    Args:
        audio: Input audio signal
        chunk_size: Size of each chunk in samples
        overlap: Overlap ratio between chunks (0.0 to 0.9)
    
    Returns:
        List of audio chunks
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1")
    
    step_size = int(chunk_size * (1 - overlap))
    chunks = []
    
    for start in range(0, len(audio), step_size):
        end = start + chunk_size
        chunk = audio[start:end]
        
        # Only include chunks that meet minimum size
        if len(chunk) >= chunk_size * 0.5:  # At least 50% of chunk_size
            # Pad if necessary
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            chunks.append(chunk)
    
    return chunks


def validate_emotion_scores(scores: Dict[str, float]) -> bool:
    """
    Validate emotion scores are in valid range.
    
    Args:
        scores: Dictionary of emotion scores
    
    Returns:
        True if valid, False otherwise
    """
    from src.config import EMOTION_LABELS
    
    if not all(emotion in scores for emotion in EMOTION_LABELS):
        return False
    
    if not all(0 <= score <= 1 for score in scores.values()):
        return False
    
    return True


def format_processing_result(
    emotions: Dict[str, float],
    audio_quality: Dict[str, float],
    suggestions: list[str],
    transcript: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format results according to ProcessingResult interface.
    
    Args:
        emotions: Emotion scores
        audio_quality: Audio quality metrics
        suggestions: Generated suggestions
        transcript: Speech transcript (mock for this module)
        metadata: Additional metadata
    
    Returns:
        Formatted result dictionary
    """
    result = {
        "transcript": transcript,
        "emotions": emotions,
        "audio_quality": audio_quality,
        "suggestions": suggestions,
        "metadata": metadata or {}
    }
    
    return result


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics for multi-label prediction.
    
    Args:
        y_true: True labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        f1_score, 
        accuracy_score, 
        precision_score, 
        recall_score,
        hamming_loss
    )
    
    # Convert predictions to binary
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = y_true.astype(int)
    
    # ===== FIX: Calculate per-label accuracy (CORRECT METRIC) =====
    # This is what you should report!
    per_label_accuracy = []
    for i in range(y_true_binary.shape[1]):
        acc = accuracy_score(y_true_binary[:, i], y_pred_binary[:, i])
        per_label_accuracy.append(acc)
    
    # Average accuracy across all emotions
    accuracy = np.mean(per_label_accuracy)
    # ==============================================================
    
    # Exact match accuracy (for reference only - too harsh!)
    exact_match = np.mean(np.all(y_pred_binary == y_true_binary, axis=1))
    
    # Hamming loss (percentage of wrong labels)
    hamming = hamming_loss(y_true_binary, y_pred_binary)
    
    # F1 scores
    f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    
    # Per-label F1 scores
    f1_per_label = f1_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
    
    # Per-label precision and recall
    precision_per_label = precision_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
    recall_per_label = recall_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
    
    metrics = {
        # Primary metrics (use these!)
        "accuracy": float(accuracy),  # ← THIS IS THE CORRECT ONE
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        
        # Secondary metrics
        "exact_match": float(exact_match),  # For reference only
        "hamming_loss": float(hamming),
        
        # Per-label details
        "per_label_accuracy": [float(x) for x in per_label_accuracy],
        "f1_per_label": [float(x) for x in f1_per_label],
        "precision_per_label": [float(x) for x in precision_per_label],
        "recall_per_label": [float(x) for x in recall_per_label],
    }
    
    return metrics

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)