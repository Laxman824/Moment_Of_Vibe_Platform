"""
Neural network models for emotion classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import numpy as np

from src.config import MODEL_CONFIG, NUM_EMOTIONS, EMOTION_LABELS

logger = logging.getLogger(__name__)


class EmotionMLP(nn.Module):
    """
    Multi-Layer Perceptron for emotion classification.
    Takes OpenSmile features as input and outputs emotion scores.
    """
    
    def __init__(
        self,
        input_size: int = MODEL_CONFIG["input_size"],
        hidden_sizes: list[int] = MODEL_CONFIG["hidden_sizes"],
        output_size: int = MODEL_CONFIG["output_size"],
        dropout: float = MODEL_CONFIG["dropout"]
    ):
        """
        Initialize MLP model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output emotions
            dropout: Dropout probability
        """
        super(EmotionMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer with sigmoid activation
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Move sigmoid to model architecture
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized EmotionMLP: {input_size} -> {hidden_sizes} -> {output_size}")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_size)
        
        Returns:
            Emotion scores (batch_size, output_size) in range [0, 1]
        """
        # Ensure input is at least 2D for BatchNorm
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)  # Sigmoid is now part of the network
    
    def predict(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Predict emotion scores for single input.
        
        Args:
            x: Input features (input_size,)
        
        Returns:
            Dictionary mapping emotion names to scores
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            
            scores = self.forward(x).squeeze().cpu().numpy()
            
            return {
                emotion: float(score)
                for emotion, score in zip(EMOTION_LABELS, scores)
            }


class EmotionCNN(nn.Module):
    """
    CNN model for emotion classification from spectrograms.
    Bonus implementation for improved accuracy.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = NUM_EMOTIONS,
        dropout: float = 0.3
    ):
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of input channels (1 for mono)
            output_size: Number of output emotions
            dropout: Dropout probability
        """
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Will be set dynamically based on input size
        self.fc_input_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(128, output_size)
        
        logger.info("Initialized EmotionCNN")
    
    def _init_fc_layer(self, x: torch.Tensor):
        """Initialize fully connected layer based on input size."""
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.fc_input_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.fc_input_size, 128).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input spectrogram (batch_size, channels, height, width)
        
        Returns:
            Emotion scores (batch_size, output_size)
        """
        # Initialize FC layer on first forward pass
        if self.fc1 is None:
            self._init_fc_layer(x)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)


class HybridEmotionModel(nn.Module):
    """
    Hybrid model combining handcrafted features (MLP) and spectrograms (CNN).
    Bonus implementation for maximum accuracy.
    """
    
    def __init__(
        self,
        feature_input_size: int = 88,
        output_size: int = NUM_EMOTIONS
    ):
        """Initialize hybrid model."""
        super(HybridEmotionModel, self).__init__()
        
        # MLP branch for handcrafted features
        self.mlp = EmotionMLP(
            input_size=feature_input_size,
            hidden_sizes=[128, 64],
            output_size=32,
            dropout=0.3
        )
        
        # CNN branch for spectrograms
        self.cnn = EmotionCNN(output_size=32, dropout=0.3)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_size)
        )
        
        logger.info("Initialized HybridEmotionModel")
    
    def forward(
        self,
        features: torch.Tensor,
        spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with both inputs.
        
        Args:
            features: Handcrafted features
            spectrogram: Spectrogram input
        
        Returns:
            Emotion scores
        """
        # Get embeddings from both branches
        mlp_out = self.mlp.network(features)
        cnn_out = self.cnn(spectrogram)
        
        # Concatenate and fuse
        combined = torch.cat([mlp_out, cnn_out], dim=1)
        output = self.fusion(combined)
        
        return torch.sigmoid(output)