"""
Script to prepare dataset for training.
Downloads and processes emotional speech datasets.
"""
import sys
from pathlib import Path
# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    EMOTION_LABELS,
    TRAINING_CONFIG,
)
from src.features import FeatureExtractor
from src.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_ravdess_filename(filename: str) -> dict:
    """
    Parse RAVDESS filename to extract emotion label.
    
    RAVDESS filename format:
    Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
    
    Emotion codes:
    01 = neutral, 02 = calm, 03 = happy, 04 = sad,
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    parts = Path(filename).stem.split('-')
    
    if len(parts) < 3:
        return None
    
    emotion_code = int(parts[2])
    
    # Map RAVDESS emotions to our labels
    emotion_mapping = {
        3: "joy",  # happy
        5: "anger",  # angry
        # Map other emotions based on energy/confidence
        2: "confidence",  # calm -> confidence
        4: "energy",  # sad -> low energy (inverse)
        6: "energy",  # fearful -> energy
        8: "joy",  # surprised -> joy
    }
    
    if emotion_code in emotion_mapping:
        return {"emotion": emotion_mapping[emotion_code]}
    
    return None


def create_multi_label(emotion: str) -> np.ndarray:
    """
    Create multi-label encoding for emotion.
    Can be extended for more sophisticated mapping.
    
    Args:
        emotion: Primary emotion label
    
    Returns:
        Multi-label array [anger, joy, energy, confidence]
    """
    # Initialize all zeros
    labels = np.zeros(len(EMOTION_LABELS))
    
    # Set primary emotion
    if emotion in EMOTION_LABELS:
        idx = EMOTION_LABELS.index(emotion)
        labels[idx] = 1.0
    
    # Add secondary correlations (domain knowledge)
    # Example: anger often correlates with high energy
    if emotion == "anger":
        labels[EMOTION_LABELS.index("energy")] = 0.7
    elif emotion == "joy":
        labels[EMOTION_LABELS.index("energy")] = 0.6
        labels[EMOTION_LABELS.index("confidence")] = 0.5
    
    return labels


def prepare_dataset(
    data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    dataset_name: str = "RAVDESS"
) -> None:
    """
    Prepare dataset by extracting features and creating splits.
    
    Args:
        data_dir: Directory containing raw audio files
        output_dir: Directory to save processed data
        dataset_name: Name of the dataset
    """
    logger.info(f"Preparing {dataset_name} dataset from {data_dir}")
    
    # Find all audio files
    audio_files = list(data_dir.rglob("*.wav"))
    
    if not audio_files:
        logger.error(f"No audio files found in {data_dir}")
        logger.info("Please download RAVDESS dataset and place in data/raw/")
        logger.info("Download from: https://zenodo.org/record/1188976")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Process files
    features_list = []
    labels_list = []
    metadata_list = []
    
    for audio_path in audio_files:
        # Parse metadata
        if dataset_name == "RAVDESS":
            metadata = parse_ravdess_filename(audio_path.name)
        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue
        
        if metadata is None:
            continue
        
        # Extract features
        features = extractor.extract_from_file(audio_path)
        
        if features is None:
            logger.warning(f"Failed to extract features from {audio_path}")
            continue
        
        # Create label
        label = create_multi_label(metadata["emotion"])
        
        features_list.append(features)
        labels_list.append(label)
        metadata_list.append({
            "filename": audio_path.name,
            "emotion": metadata["emotion"],
            "path": str(audio_path)
        })
    
    if not features_list:
        logger.error("No features extracted!")
        return
    
    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    logger.info(f"Extracted features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Create train/val/test splits
    set_seed(TRAINING_CONFIG["random_seed"])
    
    val_size = TRAINING_CONFIG["validation_split"]
    test_size = TRAINING_CONFIG["test_split"]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, metadata_list,
        test_size=test_size,
        random_state=TRAINING_CONFIG["random_seed"]
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=val_ratio,
        random_state=TRAINING_CONFIG["random_seed"]
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    # Save metadata
    with open(output_dir / "metadata_train.pkl", "wb") as f:
        pickle.dump(meta_train, f)
    with open(output_dir / "metadata_val.pkl", "wb") as f:
        pickle.dump(meta_val, f)
    with open(output_dir / "metadata_test.pkl", "wb") as f:
        pickle.dump(meta_test, f)
    
    logger.info(f"Saved processed data to {output_dir}")
    
    # Print label distribution
    logger.info("\nLabel distribution in training set:")
    for i, emotion in enumerate(EMOTION_LABELS):
        count = np.sum(y_train[:, i] > 0.5)
        logger.info(f"  {emotion}: {count} ({count/len(y_train)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prepare emotion dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="RAVDESS",
        help="Dataset name (RAVDESS, IEMOCAP, etc.)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()