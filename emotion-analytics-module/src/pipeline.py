# """
# End-to-end emotion analytics pipeline.
# Integrates feature extraction, quality analysis, and model inference.
# """
# import torch
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Optional, Any, Tuple
# import logging
# import time

# from src.config import (
#     EMOTION_LABELS,
#     SUGGESTION_RULES,
#     PROCESSING_TIME_THRESHOLD,
#     SAMPLE_RATE,
# )
# from src.features import FeatureExtractor
# from src.quality import AudioQualityAnalyzer
# from src.model import EmotionMLP
# from src.utils import load_audio, format_processing_result, timer

# logger = logging.getLogger(__name__)


# class EmotionAnalyticsPipeline:
#     """
#     Complete pipeline for emotion analytics.
#     Processes audio input and returns structured results.
#     """
    
#     def __init__(
#         self,
#         model_path: Optional[Path | str] = None,
#         device: str = "cpu"
#     ):
#         """
#         Initialize pipeline components.
        
#         Args:
#             model_path: Path to trained model checkpoint
#             device: Device for inference ('cpu' or 'cuda')
#         """
#         self.device = device
        
#         # Initialize components
#         self.feature_extractor = FeatureExtractor()
#         self.quality_analyzer = AudioQualityAnalyzer()
        
#         # Load model
#         self.model = None
#         if model_path is not None:
#             self.load_model(model_path)
        
#         logger.info(f"Initialized EmotionAnalyticsPipeline on {device}")
    
#     def load_model(self, model_path: Path | str):
#         """
#         Load trained model from checkpoint.
        
#         Args:
#             model_path: Path to model file
#         """
#         model_path = Path(model_path)
        
#         if not model_path.exists():
#             raise FileNotFoundError(f"Model not found: {model_path}")
        
#         try:
#             # Load checkpoint
#             checkpoint = torch.load(model_path, map_location=self.device)
            
#             # Initialize model architecture
#             self.model = EmotionMLP(
#                 input_size=checkpoint.get("input_size", 88),
#                 hidden_sizes=checkpoint.get("hidden_sizes", [128, 64]),
#                 output_size=checkpoint.get("output_size", 4),
#                 dropout=0.0  # No dropout during inference
#             )
            
#             # Load weights
#             self.model.load_state_dict(checkpoint["model_state_dict"])
#             self.model.to(self.device)
#             self.model.eval()
            
#             logger.info(f"Loaded model from {model_path}")
            
#         except Exception as e:
#             logger.error(f"Error loading model: {e}")
#             raise
    
#     @timer
#     def process_audio(
#         self,
#         audio_path: Path | str,
#         return_metadata: bool = True
#     ) -> Dict[str, Any]:
#         """
#         Process audio file and return emotion analytics.
#         Main entry point for the pipeline.
        
#         Args:
#             audio_path: Path to audio file
#             return_metadata: Whether to include metadata in results
        
#         Returns:
#             ProcessingResult dictionary with emotions, quality, suggestions
#         """
#         start_time = time.time()
        
#         try:
#             # Load audio
#             audio, sr = load_audio(audio_path, target_sr=SAMPLE_RATE)
            
#             # Analyze quality
#             quality_metrics = self.quality_analyzer.analyze(audio, sr)
#             quality_level, quality_issues = self.quality_analyzer.get_quality_assessment(
#                 quality_metrics
#             )
            
#             # Extract features
#             features = self.feature_extractor.extract_from_signal(audio, sr)
            
#             if features is None:
#                 raise ValueError("Feature extraction failed")
            
#             # Predict emotions
#             if self.model is None:
#                 # Return mock predictions if no model loaded
#                 logger.warning("No model loaded, returning mock predictions")
#                 emotions = self._get_mock_predictions()
#             else:
#                 emotions = self._predict_emotions(features)
            
#             # Generate suggestions
#             suggestions = self._generate_suggestions(emotions, quality_issues)
            
#             # Compute processing time
#             processing_time = time.time() - start_time
            
#             # Format metadata
#             metadata = {}
#             if return_metadata:
#                 metadata = {
#                     "processing_time_seconds": processing_time,
#                     "audio_duration_seconds": len(audio) / sr,
#                     "sample_rate": sr,
#                     "quality_level": quality_level,
#                     "quality_issues": quality_issues,
#                     "feature_count": len(features),
#                 }
            
#             # Check performance requirement
#             if processing_time > PROCESSING_TIME_THRESHOLD:
#                 logger.warning(
#                     f"Processing time {processing_time:.2f}s exceeds threshold "
#                     f"{PROCESSING_TIME_THRESHOLD}s"
#                 )
            
#             # Format result
#             result = format_processing_result(
#                 emotions=emotions,
#                 audio_quality={
#                     "snr_db": quality_metrics["snr_db"],
#                     "rms_energy": quality_metrics["rms_energy"],
#                     "quality_score": quality_metrics["quality_score"],
#                     "quality_level": quality_level,
#                 },
#                 suggestions=suggestions,
#                 transcript="",  # Mock empty transcript
#                 metadata=metadata
#             )
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing audio {audio_path}: {e}")
#             raise
    
#     def _predict_emotions(self, features: np.ndarray) -> Dict[str, float]:
#         """
#         Predict emotion scores from features.
        
#         Args:
#             features: Feature vector
        
#         Returns:
#             Dictionary of emotion scores
#         """
#         # Convert to tensor
#         features_tensor = torch.FloatTensor(features).to(self.device)
        
#         # Predict
#         self.model.eval()
#         with torch.no_grad():
#             predictions = self.model.predict(features_tensor)
        
#         return predictions
    
#     def _get_mock_predictions(self) -> Dict[str, float]:
#         """Return mock emotion predictions for testing."""
#         return {
#             "anger": 0.15,
#             "joy": 0.65,
#             "energy": 0.55,
#             "confidence": 0.70,
#         }
    
#     def _generate_suggestions(
#         self,
#         emotions: Dict[str, float],
#         quality_issues: List[str]
#     ) -> List[str]:
#         """
#         Generate context-aware suggestions based on emotions and quality.
        
#         Args:
#             emotions: Emotion scores
#             quality_issues: List of quality issues
        
#         Returns:
#             List of suggestion strings
#         """
#         suggestions = []
        
#         # Quality-based suggestions
#         if quality_issues:
#             suggestions.append(
#                 "Audio quality issues detected. Consider finding a quieter environment."
#             )
        
#         # Emotion-based suggestions (rule-based)
#         for emotion, score in emotions.items():
#             if emotion in SUGGESTION_RULES:
#                 rules = SUGGESTION_RULES[emotion]
#                 threshold = rules["threshold"]
                
#                 if emotion in ["anger", "energy", "confidence"]:
#                     # High threshold emotions
#                     if score >= threshold and "high" in rules:
#                         suggestions.append(rules["high"])
#                     elif score < threshold and "low" in rules:
#                         suggestions.append(rules["low"])
#                 else:
#                     # Joy - only trigger on high
#                     if score >= threshold and "high" in rules:
#                         suggestions.append(rules["high"])
        
#         # Combination suggestions
#         if emotions["anger"] > 0.6 and emotions["energy"] > 0.7:
#             suggestions.append(
#                 "High intensity detected. A brief pause might help both parties."
#             )
        
#         if emotions["confidence"] < 0.3 and emotions["joy"] < 0.3:
#             suggestions.append(
#                 "Conversation energy seems low. Try discussing shared interests."
#             )
        
#         # Remove duplicates
#         suggestions = list(dict.fromkeys(suggestions))
        
#         return suggestions
    
#     def process_batch(
#         self,
#         audio_paths: List[Path | str],
#         show_progress: bool = True
#     ) -> List[Dict[str, Any]]:
#         """
#         Process multiple audio files.
        
#         Args:
#             audio_paths: List of audio file paths
#             show_progress: Whether to show progress bar
        
#         Returns:
#             List of processing results
#         """
#         results = []
        
#         if show_progress:
#             from tqdm import tqdm
#             audio_paths = tqdm(audio_paths, desc="Processing audio files")
        
#         for audio_path in audio_paths:
#             try:
#                 result = self.process_audio(audio_path, return_metadata=True)
#                 results.append(result)
#             except Exception as e:
#                 logger.error(f"Error processing {audio_path}: {e}")
#                 results.append(None)
        
#         return results
    
#     def validate_performance(
#         self,
#         test_audio_paths: List[Path | str]
#     ) -> Dict[str, float]:
#         """
#         Validate pipeline performance metrics.
        
#         Args:
#             test_audio_paths: List of test audio files
        
#         Returns:
#             Performance metrics
#         """
#         processing_times = []
        
#         for audio_path in test_audio_paths:
#             start = time.time()
#             try:
#                 result = self.process_audio(audio_path, return_metadata=False)
#                 elapsed = time.time() - start
#                 processing_times.append(elapsed)
#             except Exception as e:
#                 logger.error(f"Validation error on {audio_path}: {e}")
        
#         if not processing_times:
#             return {"error": "No successful processing"}
        
#         metrics = {
#             "mean_processing_time": np.mean(processing_times),
#             "median_processing_time": np.median(processing_times),
#             "max_processing_time": np.max(processing_times),
#             "meets_latency_requirement": np.mean(processing_times) < PROCESSING_TIME_THRESHOLD,
#         }
        
#         return metrics


"""
End-to-end emotion analytics pipeline.
Integrates feature extraction, quality analysis, and model inference.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import pickle

from src.config import (
    EMOTION_LABELS,
    SUGGESTION_RULES,
    PROCESSING_TIME_THRESHOLD,
    SAMPLE_RATE,
)
from src.features import FeatureExtractor
from src.quality import AudioQualityAnalyzer
from src.model import EmotionMLP
from src.utils import load_audio, format_processing_result, timer

logger = logging.getLogger(__name__)


class EmotionAnalyticsPipeline:
    """
    Complete pipeline for emotion analytics.
    Processes audio input and returns structured results.
    """
    
    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        device: str = "cpu"
    ):
        """
        Initialize pipeline components.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.quality_analyzer = AudioQualityAnalyzer()
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        
        if model_path is not None:
            self.load_model(model_path)
        
        logger.info(f"Initialized EmotionAnalyticsPipeline on {device}")
    
    def load_model(self, model_path: Path | str):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize model architecture
            self.model = EmotionMLP(
                input_size=checkpoint.get("input_size", 88),
                hidden_sizes=checkpoint.get("hidden_sizes", [128, 64]),
                output_size=checkpoint.get("output_size", 4),
                dropout=0.0  # No dropout during inference
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded model from {model_path}")
            
            # Load scaler if it exists
            scaler_path = model_path.parent / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.warning(f"No scaler found at {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @timer
    def process_audio(
        self,
        audio_path: Path | str,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio file and return emotion analytics.
        Main entry point for the pipeline.
        
        Args:
            audio_path: Path to audio file
            return_metadata: Whether to include metadata in results
        
        Returns:
            ProcessingResult dictionary with emotions, quality, suggestions
        """
        start_time = time.time()
        
        try:
            # Load audio
            audio, sr = load_audio(audio_path, target_sr=SAMPLE_RATE)
            
            # Analyze quality
            quality_metrics = self.quality_analyzer.analyze(audio, sr)
            quality_level, quality_issues = self.quality_analyzer.get_quality_assessment(
                quality_metrics
            )
            
            # Extract features
            features = self.feature_extractor.extract_from_signal(audio, sr)
            
            if features is None:
                raise ValueError("Feature extraction failed")
            
            # Predict emotions
            if self.model is None:
                # Return mock predictions if no model loaded
                logger.warning("No model loaded, returning mock predictions")
                emotions = self._get_mock_predictions()
            else:
                emotions = self._predict_emotions(features)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(emotions, quality_issues)
            
            # Compute processing time
            processing_time = time.time() - start_time
            
            # Format metadata
            metadata = {}
            if return_metadata:
                metadata = {
                    "processing_time_seconds": processing_time,
                    "audio_duration_seconds": len(audio) / sr,
                    "sample_rate": sr,
                    "quality_level": quality_level,
                    "quality_issues": quality_issues,
                    "feature_count": len(features),
                }
            
            # Check performance requirement
            if processing_time > PROCESSING_TIME_THRESHOLD:
                logger.warning(
                    f"Processing time {processing_time:.2f}s exceeds threshold "
                    f"{PROCESSING_TIME_THRESHOLD}s"
                )
            
            # Format result
            result = format_processing_result(
                emotions=emotions,
                audio_quality={
                    "snr_db": quality_metrics["snr_db"],
                    "rms_energy": quality_metrics["rms_energy"],
                    "quality_score": quality_metrics["quality_score"],
                    "quality_level": quality_level,
                },
                suggestions=suggestions,
                transcript="",  # Mock empty transcript
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            raise
    
    def _predict_emotions(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion scores from features.
        
        Args:
            features: Feature vector
        
        Returns:
            Dictionary of emotion scores
        """
        # Normalize features if scaler exists
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)  # Forward pass
            
            # Get probabilities (model has sigmoid activation already)
            probs = outputs.cpu().numpy().flatten()
        
        # Map to emotion labels based on model output
        # Assuming output is [anger, joy, energy, confidence] or similar
        emotions = {
            "anger": float(probs[0]) if len(probs) > 0 else 0.0,
            "joy": float(probs[1]) if len(probs) > 1 else 0.0,
            "energy": float(probs[2]) if len(probs) > 2 else 0.0,
            "confidence": float(probs[3]) if len(probs) > 3 else 0.0,
        }
        
        logger.debug(f"Predicted emotions: {emotions}")
        
        return emotions
    
    def _get_mock_predictions(self) -> Dict[str, float]:
        """Return mock emotion predictions for testing."""
        return {
            "anger": 0.15,
            "joy": 0.65,
            "energy": 0.55,
            "confidence": 0.70,
        }
    
    def _generate_suggestions(
        self,
        emotions: Dict[str, float],
        quality_issues: List[str]
    ) -> List[str]:
        """
        Generate context-aware suggestions based on emotions and quality.
        
        Args:
            emotions: Emotion scores
            quality_issues: List of quality issues
        
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Quality-based suggestions
        if quality_issues:
            suggestions.append(
                "Audio quality issues detected. Consider finding a quieter environment."
            )
        
        # Emotion-based suggestions (rule-based)
        for emotion, score in emotions.items():
            if emotion in SUGGESTION_RULES:
                rules = SUGGESTION_RULES[emotion]
                threshold = rules["threshold"]
                
                if emotion in ["anger", "energy", "confidence"]:
                    # High threshold emotions
                    if score >= threshold and "high" in rules:
                        suggestions.append(rules["high"])
                    elif score < threshold and "low" in rules:
                        suggestions.append(rules["low"])
                else:
                    # Joy - only trigger on high
                    if score >= threshold and "high" in rules:
                        suggestions.append(rules["high"])
        
        # Combination suggestions
        if emotions["anger"] > 0.6 and emotions["energy"] > 0.7:
            suggestions.append(
                "High intensity detected. A brief pause might help both parties."
            )
        
        if emotions["confidence"] < 0.3 and emotions["joy"] < 0.3:
            suggestions.append(
                "Conversation energy seems low. Try discussing shared interests."
            )
        
        # Remove duplicates
        suggestions = list(dict.fromkeys(suggestions))
        
        return suggestions
    
    def process_batch(
        self,
        audio_paths: List[Path | str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            show_progress: Whether to show progress bar
        
        Returns:
            List of processing results
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            audio_paths = tqdm(audio_paths, desc="Processing audio files")
        
        for audio_path in audio_paths:
            try:
                result = self.process_audio(audio_path, return_metadata=True)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                results.append(None)
        
        return results
    
    def validate_performance(
        self,
        test_audio_paths: List[Path | str]
    ) -> Dict[str, float]:
        """
        Validate pipeline performance metrics.
        
        Args:
            test_audio_paths: List of test audio files
        
        Returns:
            Performance metrics
        """
        processing_times = []
        
        for audio_path in test_audio_paths:
            start = time.time()
            try:
                result = self.process_audio(audio_path, return_metadata=False)
                elapsed = time.time() - start
                processing_times.append(elapsed)
            except Exception as e:
                logger.error(f"Validation error on {audio_path}: {e}")
        
        if not processing_times:
            return {"error": "No successful processing"}
        
        metrics = {
            "mean_processing_time": np.mean(processing_times),
            "median_processing_time": np.median(processing_times),
            "max_processing_time": np.max(processing_times),
            "meets_latency_requirement": np.mean(processing_times) < PROCESSING_TIME_THRESHOLD,
        }
        
        return metrics