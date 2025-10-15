"""
Tests for emotion classification models.
"""
import pytest
import torch
import numpy as np

from src.model import EmotionMLP, EmotionCNN
from src.config import NUM_EMOTIONS


class TestEmotionMLP:
    """Test cases for EmotionMLP."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = EmotionMLP(input_size=88, hidden_sizes=[128, 64], output_size=4)
        assert model is not None
        assert model.input_size == 88
        assert model.output_size == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = EmotionMLP(input_size=88, hidden_sizes=[128, 64], output_size=4)
        batch_size = 16
        
        x = torch.randn(batch_size, 88)
        output = model(x)
        
        assert output.shape == (batch_size, 4)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
    
    def test_predict(self):
        """Test prediction method."""
        model = EmotionMLP(input_size=88, hidden_sizes=[128, 64], output_size=4)
        
        x = torch.randn(88)
        predictions = model.predict(x)
        
        assert isinstance(predictions, dict)
        assert len(predictions) == NUM_EMOTIONS
        assert all(0 <= score <= 1 for score in predictions.values())
    
    def test_single_sample_prediction(self):
        """Test prediction on single sample."""
        model = EmotionMLP(input_size=88)
        x = torch.randn(88)
        
        predictions = model.predict(x)
        
        assert "anger" in predictions
        assert "joy" in predictions
        assert "energy" in predictions
        assert "confidence" in predictions


class TestEmotionCNN:
    """Test cases for EmotionCNN."""
    
    def test_initialization(self):
        """Test CNN initialization."""
        model = EmotionCNN(input_channels=1, output_size=4)
        assert model is not None
    
    def test_forward_pass(self):
        """Test CNN forward pass."""
        model = EmotionCNN(input_channels=1, output_size=4)
        batch_size = 8
        
        # Dummy spectrogram input (batch, channels, height, width)
        x = torch.randn(batch_size, 1, 128, 128)
        output = model(x)
        
        assert output.shape == (batch_size, 4)
        assert torch.all((output >= 0) & (output <= 1))