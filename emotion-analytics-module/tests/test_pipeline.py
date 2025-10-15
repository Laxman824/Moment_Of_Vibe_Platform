"""
Tests for emotion analytics pipeline.
"""
import pytest
import numpy as np
from pathlib import Path
import soundfile as sf
import tempfile
import torch

from src.pipeline import EmotionAnalyticsPipeline
from src.model import EmotionMLP
from src.config import SAMPLE_RATE, CHECKPOINTS_DIR


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file."""
    duration = 5
    sr = SAMPLE_RATE
    t = np.linspace(0, duration, int(duration * sr))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        audio_path = Path(f.name)
    
    yield audio_path
    
    audio_path.unlink()


@pytest.fixture
def mock_model():
    """Create a mock trained model."""
    model = EmotionMLP(input_size=88, hidden_sizes=[128, 64], output_size=4)
    
    # Save model
    model_path = CHECKPOINTS_DIR / "test_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": 88,
        "hidden_sizes": [128, 64],
        "output_size": 4,
    }, model_path)
    
    yield model_path
    
    model_path.unlink()


class TestEmotionAnalyticsPipeline:
    """Test cases for EmotionAnalyticsPipeline."""
    
    def test_initialization_without_model(self):
        """Test pipeline initialization without model."""
        pipeline = EmotionAnalyticsPipeline()
        assert pipeline is not None
        assert pipeline.model is None
    
    def test_initialization_with_model(self, mock_model):
        """Test pipeline initialization with model."""
        pipeline = EmotionAnalyticsPipeline(model_path=mock_model)
        assert pipeline is not None
        assert pipeline.model is not None
    
    def test_process_audio_without_model(self, sample_audio_file):
        """Test processing without loaded model (mock predictions)."""
        pipeline = EmotionAnalyticsPipeline()
        
        result = pipeline.process_audio(sample_audio_file)
        
        assert result is not None
        assert "emotions" in result
        assert "audio_quality" in result
        assert "suggestions" in result
        assert "transcript" in result
        assert "metadata" in result
    
    def test_process_audio_with_model(self, sample_audio_file, mock_model):
        """Test processing with loaded model."""
        pipeline = EmotionAnalyticsPipeline(model_path=mock_model)
        
        result = pipeline.process_audio(sample_audio_file)
        
        assert result is not None
        assert len(result["emotions"]) == 4
        assert all(0 <= score <= 1 for score in result["emotions"].values())
    
    def test_result_structure(self, sample_audio_file):
        """Test output structure matches ProcessingResult schema."""
        pipeline = EmotionAnalyticsPipeline()
        
        result = pipeline.process_audio(sample_audio_file)
        
        # Check required fields
        assert isinstance(result["transcript"], str)
        assert isinstance(result["emotions"], dict)
        assert isinstance(result["audio_quality"], dict)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["metadata"], dict)
        
        # Check emotion fields
        assert "anger" in result["emotions"]
        assert "joy" in result["emotions"]
        assert "energy" in result["emotions"]
        assert "confidence" in result["emotions"]
    
    def test_suggestions_generation(self, sample_audio_file):
        """Test suggestion generation."""
        pipeline = EmotionAnalyticsPipeline()
        
        result = pipeline.process_audio(sample_audio_file)
        
        assert isinstance(result["suggestions"], list)
    
    def test_process_batch(self, sample_audio_file):
        """Test batch processing."""
        pipeline = EmotionAnalyticsPipeline()
        
        results = pipeline.process_batch(
            [sample_audio_file, sample_audio_file],
            show_progress=False
        )
        
        assert len(results) == 2
        assert all(r is not None for r in results)
    
    def test_processing_time(self, sample_audio_file):
        """Test processing time requirement."""
        pipeline = EmotionAnalyticsPipeline()
        
        result = pipeline.process_audio(sample_audio_file)
        
        processing_time = result["metadata"]["processing_time_seconds"]
        assert processing_time < 1.5  # Requirement: <1.5s per 10s chunk