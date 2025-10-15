"""
Tests for utility functions.
"""
import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
import soundfile as sf

from src.utils import (
    timer,
    load_audio,
    resample_audio,
    chunk_audio,
    validate_emotion_scores,
    format_processing_result,
    calculate_metrics,
    set_seed,
)


class TestTimer:
    """Test timer decorator."""
    
    def test_timer_decorator(self, caplog):
        """Test that timer decorator logs execution time."""
        import logging
        caplog.set_level(logging.INFO)
        
        @timer
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        result = slow_function()
        assert result == "done"
        # Check that some timing info was logged
        assert len(caplog.records) > 0


class TestLoadAudio:
    """Test audio loading functionality."""
    
    def test_load_audio_mono(self, tmp_path):
        """Test loading mono audio file."""
        audio_path = tmp_path / "test_mono.wav"
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        sf.write(audio_path, audio, sr)
        
        loaded_audio, loaded_sr = load_audio(audio_path)
        
        assert loaded_sr == sr
        assert len(loaded_audio) == len(audio)
        assert np.max(np.abs(loaded_audio)) <= 1.0
    
    def test_load_audio_stereo_to_mono(self, tmp_path):
        """Test loading stereo audio and converting to mono."""
        audio_path = tmp_path / "test_stereo.wav"
        sr = 16000
        stereo_audio = np.random.randn(sr, 2).astype(np.float32)
        sf.write(audio_path, stereo_audio, sr)
        
        loaded_audio, loaded_sr = load_audio(audio_path)
        
        assert loaded_sr == sr
        assert loaded_audio.ndim == 1
    
    def test_load_audio_with_resampling(self, tmp_path):
        """Test audio loading with resampling."""
        audio_path = tmp_path / "test_resample.wav"
        orig_sr = 44100
        target_sr = 16000
        audio = np.random.randn(orig_sr).astype(np.float32)
        sf.write(audio_path, audio, orig_sr)
        
        loaded_audio, loaded_sr = load_audio(audio_path, target_sr=target_sr)
        
        assert loaded_sr == target_sr
        expected_length = int(len(audio) * target_sr / orig_sr)
        assert abs(len(loaded_audio) - expected_length) < 100
    
    def test_load_audio_without_normalization(self, tmp_path):
        """Test loading audio without normalization."""
        audio_path = tmp_path / "test_no_norm.wav"
        sr = 16000
        # Create audio with max value of 0.5
        audio = np.ones(sr).astype(np.float32) * 0.5
        sf.write(audio_path, audio, sr)
        
        loaded_audio, loaded_sr = load_audio(audio_path, normalize=False)
        
        assert loaded_sr == sr
        # Max should still be around 0.5, not normalized to 1.0
        assert np.max(np.abs(loaded_audio)) < 0.6
    
    def test_load_audio_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file.wav")
    
    def test_load_audio_invalid_format(self, tmp_path):
        """Test error handling for invalid audio format."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not audio data")
        
        with pytest.raises(ValueError):
            load_audio(invalid_file)


class TestResampleAudio:
    """Test audio resampling."""
    
    def test_resample_audio_same_rate(self):
        """Test resampling with same rate (no change)."""
        audio = np.random.randn(16000)
        resampled = resample_audio(audio, 16000, 16000)
        
        assert np.array_equal(audio, resampled)
    
    def test_resample_audio_downsample(self):
        """Test downsampling audio."""
        orig_sr = 44100
        target_sr = 16000
        duration = 1.0
        audio = np.random.randn(int(orig_sr * duration))
        
        resampled = resample_audio(audio, orig_sr, target_sr)
        
        expected_length = int(len(audio) * target_sr / orig_sr)
        assert abs(len(resampled) - expected_length) < 10
    
    def test_resample_audio_upsample(self):
        """Test upsampling audio."""
        orig_sr = 8000
        target_sr = 16000
        audio = np.random.randn(8000)
        
        resampled = resample_audio(audio, orig_sr, target_sr)
        
        expected_length = int(len(audio) * target_sr / orig_sr)
        assert abs(len(resampled) - expected_length) < 10


class TestChunkAudio:
    """Test audio chunking."""
    
    def test_chunk_audio_no_overlap(self):
        """Test chunking without overlap."""
        audio = np.random.randn(16000)
        chunk_size = 4000
        
        chunks = chunk_audio(audio, chunk_size, overlap=0.0)
        
        assert len(chunks) == 4
        assert all(len(chunk) == chunk_size for chunk in chunks)
    
    def test_chunk_audio_with_overlap(self):
        """Test chunking with overlap."""
        audio = np.random.randn(16000)
        chunk_size = 4000
        overlap = 0.5
        
        chunks = chunk_audio(audio, chunk_size, overlap=overlap)
        
        assert len(chunks) > 0
        assert all(len(chunk) == chunk_size for chunk in chunks)
    
    def test_chunk_audio_padding(self):
        """Test that short chunks are padded."""
        audio = np.random.randn(10000)
        chunk_size = 4000
        
        chunks = chunk_audio(audio, chunk_size, overlap=0.0)
        
        assert all(len(chunk) == chunk_size for chunk in chunks)
    
    def test_chunk_audio_invalid_overlap(self):
        """Test error handling for invalid overlap."""
        audio = np.random.randn(16000)
        
        with pytest.raises(ValueError):
            chunk_audio(audio, 4000, overlap=-0.1)
        
        with pytest.raises(ValueError):
            chunk_audio(audio, 4000, overlap=1.0)


class TestValidateEmotionScores:
    """Test emotion score validation."""
    
    def test_validate_emotion_scores_valid(self):
        """Test validation with valid scores."""
        from src.config import EMOTION_LABELS
        
        # Create valid scores for all emotions
        scores = {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
        
        assert validate_emotion_scores(scores) is True
    
    def test_validate_emotion_scores_missing_emotion(self):
        """Test validation with missing emotion."""
        scores = {
            "neutral": 0.5,
            "calm": 0.5,
        }
        
        assert validate_emotion_scores(scores) is False
    
    def test_validate_emotion_scores_out_of_range(self):
        """Test validation with out-of-range scores."""
        from src.config import EMOTION_LABELS
        
        scores = {label: 0.1 for label in EMOTION_LABELS}
        scores[EMOTION_LABELS[0]] = 1.5  # Invalid
        
        assert validate_emotion_scores(scores) is False
    
    def test_validate_emotion_scores_negative(self):
        """Test validation with negative scores."""
        from src.config import EMOTION_LABELS
        
        scores = {label: 0.1 for label in EMOTION_LABELS}
        scores[EMOTION_LABELS[0]] = -0.1  # Invalid
        
        assert validate_emotion_scores(scores) is False


class TestFormatProcessingResult:
    """Test result formatting."""
    
    def test_format_processing_result_basic(self):
        """Test basic result formatting."""
        emotions = {"happy": 0.8, "sad": 0.2}
        quality = {"snr_db": 20.0, "quality_score": 0.9}
        suggestions = ["Good audio quality"]
        
        result = format_processing_result(emotions, quality, suggestions)
        
        assert result["emotions"] == emotions
        assert result["audio_quality"] == quality
        assert result["suggestions"] == suggestions
        assert result["transcript"] == ""
        assert "metadata" in result
    
    def test_format_processing_result_with_transcript(self):
        """Test formatting with transcript."""
        result = format_processing_result(
            emotions={"happy": 0.5},
            audio_quality={"snr_db": 15.0},
            suggestions=["test"],
            transcript="Hello world"
        )
        
        assert result["transcript"] == "Hello world"
    
    def test_format_processing_result_with_metadata(self):
        """Test formatting with metadata."""
        metadata = {"duration": 5.2, "sample_rate": 16000}
        
        result = format_processing_result(
            emotions={"happy": 0.5},
            audio_quality={"snr_db": 15.0},
            suggestions=["test"],
            metadata=metadata
        )
        
        assert result["metadata"] == metadata


class TestCalculateMetrics:
    """Test metrics calculation."""
    
    def test_calculate_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.1, 0.0], [0.0, 0.95, 0.05], [0.0, 0.1, 0.9]])
        
        metrics = calculate_metrics(y_true, y_pred, threshold=0.5)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_micro" in metrics
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] > 0.9
    
    def test_calculate_metrics_random(self):
        """Test metrics with random predictions."""
        y_true = np.random.randint(0, 2, (100, 8))
        y_pred = np.random.rand(100, 8)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert 0 <= metrics["f1_micro"] <= 1
    
    def test_calculate_metrics_threshold(self):
        """Test metrics with different threshold."""
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.6, 0.4], [0.3, 0.7]])
        
        metrics = calculate_metrics(y_true, y_pred, threshold=0.5)
        assert metrics["accuracy"] == 1.0


class TestSetSeed:
    """Test random seed setting."""
    
    def test_set_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_seed(42)
        rand1 = np.random.rand(10)
        
        set_seed(42)
        rand2 = np.random.rand(10)
        
        assert np.array_equal(rand1, rand2)
    
    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        rand1 = np.random.rand(10)
        
        set_seed(123)
        rand2 = np.random.rand(10)
        
        assert not np.array_equal(rand1, rand2)