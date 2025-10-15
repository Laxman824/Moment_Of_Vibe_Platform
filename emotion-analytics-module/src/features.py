"""
Feature extraction module using OpenSmile.
Handles acoustic feature extraction from audio signals.
"""
import numpy as np
import opensmile
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from src.config import (
    OPENSMILE_FEATURE_SET,
    OPENSMILE_FEATURE_LEVEL,
    SAMPLE_RATE,
    CHUNK_SIZE_SAMPLES,
)
from src.utils import load_audio, chunk_audio, timer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts acoustic features from audio using OpenSmile.
    Implements eGeMAPSv02 feature set for emotion-relevant features.
    """
    
    def __init__(
        self,
        feature_set: str = OPENSMILE_FEATURE_SET,
        feature_level: str = OPENSMILE_FEATURE_LEVEL,
        sample_rate: int = SAMPLE_RATE
    ):
        """
        Initialize feature extractor.
        
        Args:
            feature_set: OpenSmile feature set name
            feature_level: Feature extraction level
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.feature_set = feature_set
        self.feature_level = feature_level
        
        # Initialize OpenSmile
        try:
            if feature_set == "eGeMAPSv02":
                opensmile_feature_set = opensmile.FeatureSet.eGeMAPSv02
            elif feature_set == "GeMAPSv01b":
                opensmile_feature_set = opensmile.FeatureSet.GeMAPSv01b
            else:
                opensmile_feature_set = opensmile.FeatureSet.eGeMAPSv02
            
            if feature_level == "Functionals":
                opensmile_feature_level = opensmile.FeatureLevel.Functionals
            else:
                opensmile_feature_level = opensmile.FeatureLevel.LowLevelDescriptors
            
            self.smile = opensmile.Smile(
                feature_set=opensmile_feature_set,
                feature_level=opensmile_feature_level,
            )
            logger.info(f"Initialized OpenSmile with {feature_set} / {feature_level}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenSmile: {e}")
            raise
    
    @timer
    def extract_from_file(
        self,
        audio_path: Path | str,
        aggregate: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            aggregate: Whether to aggregate features across chunks
        
        Returns:
            Feature vector as numpy array, or None on failure
        """
        try:
            audio, sr = load_audio(audio_path, target_sr=self.sample_rate)
            return self.extract_from_signal(audio, sr, aggregate=aggregate)
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def extract_from_signal(
        self,
        audio: np.ndarray,
        sample_rate: int,
        aggregate: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract features from audio signal.
        
        Args:
            audio: Audio signal array
            sample_rate: Sample rate of audio
            aggregate: Whether to aggregate features across chunks
        
        Returns:
            Feature vector as numpy array, or None on failure
        """
        try:
            # Process entire audio if short enough, otherwise chunk
            if len(audio) <= CHUNK_SIZE_SAMPLES:
                features_df = self.smile.process_signal(audio, sample_rate)
                return features_df.values.flatten()
            
            # Process in chunks for longer audio
            chunks = chunk_audio(audio, CHUNK_SIZE_SAMPLES, overlap=0.0)
            chunk_features = []
            
            for chunk in chunks:
                features_df = self.smile.process_signal(chunk, sample_rate)
                chunk_features.append(features_df.values.flatten())
            
            if not chunk_features:
                return None
            
            # Aggregate features
            if aggregate:
                # Mean aggregation across chunks
                features = np.mean(chunk_features, axis=0)
            else:
                # Return all chunk features
                features = np.array(chunk_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from signal: {e}")
            return None
    
    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        try:
            # Create dummy signal to get feature names
            dummy_signal = np.random.randn(SAMPLE_RATE)
            features_df = self.smile.process_signal(dummy_signal, self.sample_rate)
            return features_df.columns.tolist()
        except Exception as e:
            logger.error(f"Error getting feature names: {e}")
            return []
    
    def extract_batch(
        self,
        audio_paths: list[Path | str],
        show_progress: bool = True
    ) -> list[Optional[np.ndarray]]:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            show_progress: Whether to show progress bar
        
        Returns:
            List of feature vectors
        """
        features_list = []
        
        if show_progress:
            from tqdm import tqdm
            audio_paths = tqdm(audio_paths, desc="Extracting features")
        
        for audio_path in audio_paths:
            features = self.extract_from_file(audio_path, aggregate=True)
            features_list.append(features)
        
        return features_list


class LibrosaFeatureExtractor:
    """
    Fallback feature extractor using librosa.
    Used when OpenSmile is not available or fails.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """Initialize librosa-based extractor."""
        self.sample_rate = sample_rate
        logger.info("Initialized Librosa feature extractor (fallback)")
    
    def extract_from_file(
        self,
        audio_path: Path | str,
        aggregate: bool = True
    ) -> Optional[np.ndarray]:
        """Extract MFCC and related features using librosa."""
        import librosa
        
        try:
            audio, sr = load_audio(audio_path, target_sr=self.sample_rate)
            return self.extract_from_signal(audio, sr, aggregate=aggregate)
        except Exception as e:
            logger.error(f"Error extracting librosa features from {audio_path}: {e}")
            return None
    
    def extract_from_signal(
        self,
        audio: np.ndarray,
        sample_rate: int,
        aggregate: bool = True
    ) -> Optional[np.ndarray]:
        """Extract features from audio signal using librosa."""
        import librosa
        
        try:
            # Extract multiple feature types
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Aggregate features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                [np.mean(spectral_centroid)],
                [np.std(spectral_centroid)],
                [np.mean(spectral_rolloff)],
                [np.std(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                [np.std(zero_crossing_rate)],
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting librosa features: {e}")
            return None