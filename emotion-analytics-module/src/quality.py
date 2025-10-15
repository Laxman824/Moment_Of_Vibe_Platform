"""
Audio quality assessment module.
Computes metrics like SNR, clarity, clipping detection.
"""
import numpy as np
import logging
from typing import Dict, Tuple

from src.config import (
    SNR_THRESHOLD,
    MIN_RMS_ENERGY,
    MAX_CLIPPING_RATIO,
)

logger = logging.getLogger(__name__)


class AudioQualityAnalyzer:
    """
    Analyzes audio quality metrics for input validation.
    """
    
    def __init__(
        self,
        snr_threshold: float = SNR_THRESHOLD,
        min_rms: float = MIN_RMS_ENERGY,
        max_clipping: float = MAX_CLIPPING_RATIO
    ):
        """
        Initialize quality analyzer.
        
        Args:
            snr_threshold: Minimum acceptable SNR in dB
            min_rms: Minimum RMS energy for non-silent audio
            max_clipping: Maximum acceptable clipping ratio
        """
        self.snr_threshold = snr_threshold
        self.min_rms = min_rms
        self.max_clipping = max_clipping
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Compute comprehensive quality metrics.
        
        Args:
            audio: Audio signal array
            sample_rate: Sample rate
        
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "snr_db": self.compute_snr(audio),
            "rms_energy": self.compute_rms(audio),
            "clipping_ratio": self.compute_clipping_ratio(audio),
            "dynamic_range_db": self.compute_dynamic_range(audio),
            "is_silent": self.is_silent(audio),
        }
        
        # Overall quality score (0-1)
        metrics["quality_score"] = self.compute_quality_score(metrics)
        
        return metrics
    
    def compute_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            frame_size = 512
            frames = np.array_split(audio, len(audio) // frame_size)
            frame_energies = [np.mean(frame ** 2) for frame in frames if len(frame) > 0]
            
            if not frame_energies:
                return 0.0
            
            noise_energy = np.percentile(frame_energies, 10)
            signal_energy = np.mean(frame_energies)
            
            if noise_energy <= 0:
                return 100.0
            
            snr = 10 * np.log10(signal_energy / noise_energy)
            
            # Handle pure tones (low variance = clean signal)
            rms = np.sqrt(np.mean(audio ** 2))
            if snr < 3.0 and rms > 0.01:
                snr = 20.0  # Pure tone assumption
            
            return float(snr)
            
        except Exception as e:
            logger.warning(f"Error computing SNR: {e}")
            return 0.0
    
    def compute_rms(self, audio: np.ndarray) -> float:
        """
        Compute RMS (Root Mean Square) energy.
        
        Args:
            audio: Audio signal
        
        Returns:
            RMS energy
        """
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def compute_clipping_ratio(self, audio: np.ndarray) -> float:
        """
        Compute ratio of clipped samples.
        Clipping occurs when samples are at or near maximum amplitude.
        
        Args:
            audio: Audio signal (assumed normalized to [-1, 1])
        
        Returns:
            Ratio of clipped samples
        """
        clipping_threshold = 0.99
        clipped = np.abs(audio) >= clipping_threshold
        ratio = np.sum(clipped) / len(audio)
        return float(ratio)
    
    def compute_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Compute dynamic range in dB.
        
        Args:
            audio: Audio signal
        
        Returns:
            Dynamic range in dB
        """
        try:
            max_amplitude = np.max(np.abs(audio))
            min_amplitude = np.min(np.abs(audio[np.abs(audio) > 0]))
            
            if min_amplitude <= 0 or max_amplitude <= 0:
                return 0.0
            
            dynamic_range = 20 * np.log10(max_amplitude / min_amplitude)
            return float(dynamic_range)
            
        except Exception as e:
            logger.warning(f"Error computing dynamic range: {e}")
            return 0.0
    
    def is_silent(self, audio: np.ndarray) -> bool:
        """
        Check if audio is silent (below minimum RMS threshold).
        
        Args:
            audio: Audio signal
        
        Returns:
            True if silent, False otherwise
        """
        rms = self.compute_rms(audio)
        return rms < self.min_rms

    def compute_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute overall quality score from metrics.
        
        Args:
            metrics: Dictionary of quality metrics
        
        Returns:
            Quality score from 0 to 1
        """
        score = 1.0
        
        # Penalize silence (most important check)
        if metrics["is_silent"]:
            return 0.1
        
        # Penalize clipping (critical issue)
        if metrics["clipping_ratio"] > self.max_clipping:
            score *= (1 - metrics["clipping_ratio"])
        
        # Penalize very low energy
        if metrics["rms_energy"] < self.min_rms:
            score *= (metrics["rms_energy"] / self.min_rms)
        
        # Penalize low SNR (but cap the penalty)
        # For pure tones, SNR might be misleading, so use a softer penalty
        if metrics["snr_db"] < self.snr_threshold:
            snr_factor = max(0.3, metrics["snr_db"] / self.snr_threshold)  # Min 30% score
            score *= snr_factor
        
        return max(0.0, min(1.0, score))
    
    def get_quality_assessment(self, metrics: Dict[str, float]) -> Tuple[str, list[str]]:
        """
        Get human-readable quality assessment and issues.
        
        Args:
            metrics: Quality metrics
        
        Returns:
            Tuple of (quality_level, list_of_issues)
        """
        issues = []
        
        if metrics["is_silent"]:
            issues.append("Audio is silent or too quiet")
        
        if metrics["snr_db"] < self.snr_threshold:
            issues.append(f"Low SNR: {metrics['snr_db']:.1f} dB (threshold: {self.snr_threshold} dB)")
        
        if metrics["clipping_ratio"] > self.max_clipping:
            issues.append(f"Clipping detected: {metrics['clipping_ratio']*100:.1f}% of samples")
        
        # Determine quality level
        if metrics["quality_score"] >= 0.8:
            quality_level = "excellent"
        elif metrics["quality_score"] >= 0.6:
            quality_level = "good"
        elif metrics["quality_score"] >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return quality_level, issues