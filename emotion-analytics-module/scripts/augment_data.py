#!/usr/bin/env python3
"""
Augment training data to balance classes and improve accuracy.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import logging
from src.config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def augment_features(features, augmentation_type='noise'):
    """
    Augment feature vector.
    
    Args:
        features: Original features
        augmentation_type: Type of augmentation
    
    Returns:
        Augmented features
    """
    if augmentation_type == 'noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 0.02, features.shape)
        return features + noise
    
    elif augmentation_type == 'scale':
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        return features * scale
    
    elif augmentation_type == 'shift':
        # Random shift
        shift = np.random.normal(0, 0.01, features.shape)
        return features + shift
    
    elif augmentation_type == 'mixup':
        # Mix with another random sample
        # This requires the full dataset, so return as is
        return features
    
    return features


def oversample_minority_classes(X_train, y_train, target_ratio=0.5):
    """
    Oversample minority classes to balance dataset.
    
    Args:
        X_train: Training features
        y_train: Training labels
        target_ratio: Target ratio for minority classes (0.5 = 50% of majority)
    
    Returns:
        Balanced X_train, y_train
    """
    logger.info("Analyzing class distribution...")
    
    # Calculate class frequencies
    class_counts = y_train.sum(axis=0)
    max_count = class_counts.max()
    
    logger.info(f"Original class distribution:")
    for i, count in enumerate(class_counts):
        logger.info(f"  Class {i}: {count:.0f} ({count/len(y_train)*100:.1f}%)")
    
    logger.info(f"\nTarget: {target_ratio*100:.0f}% of majority class ({max_count*target_ratio:.0f} samples)")
    
    # Collect augmented samples
    X_augmented = [X_train]
    y_augmented = [y_train]
    
    for class_idx in range(y_train.shape[1]):
        current_count = class_counts[class_idx]
        target_count = max(current_count, max_count * target_ratio)
        needed = int(target_count - current_count)
        
        if needed > 0:
            logger.info(f"\nClass {class_idx}: need {needed} additional samples")
            
            # Get samples that have this label
            class_mask = y_train[:, class_idx] == 1
            class_samples_X = X_train[class_mask]
            class_samples_y = y_train[class_mask]
            
            if len(class_samples_X) == 0:
                logger.warning(f"  No samples found for class {class_idx}, skipping")
                continue
            
            # Generate augmented samples
            for i in range(needed):
                # Random sample from this class
                idx = np.random.randint(0, len(class_samples_X))
                sample_X = class_samples_X[idx]
                sample_y = class_samples_y[idx]
                
                # Random augmentation
                aug_type = np.random.choice(['noise', 'scale', 'shift'])
                augmented_X = augment_features(sample_X, aug_type)
                
                X_augmented.append(augmented_X.reshape(1, -1))
                y_augmented.append(sample_y.reshape(1, -1))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Generated {i+1}/{needed} samples")
    
    # Combine all data
    X_balanced = np.vstack(X_augmented)
    y_balanced = np.vstack(y_augmented)
    
    # Shuffle
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Augmentation complete!")
    logger.info(f"Original: {len(X_train)} samples")
    logger.info(f"Augmented: {len(X_balanced)} samples (+{len(X_balanced)-len(X_train)})")
    logger.info(f"\nNew class distribution:")
    
    new_counts = y_balanced.sum(axis=0)
    for i, count in enumerate(new_counts):
        logger.info(f"  Class {i}: {count:.0f} ({count/len(y_balanced)*100:.1f}%)")
    logger.info(f"{'='*60}\n")
    
    return X_balanced, y_balanced


def main():
    logger.info("="*60)
    logger.info("DATA AUGMENTATION FOR CLASS BALANCING")
    logger.info("="*60 + "\n")
    
    # Load data
    logger.info("Loading training data...")
    X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    
    logger.info(f"Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features\n")
    
    # Augment data
    X_balanced, y_balanced = oversample_minority_classes(
        X_train, y_train, target_ratio=0.6  # Boost minority to 60% of majority
    )
    
    # Save augmented data
    output_dir = PROCESSED_DATA_DIR
    np.save(output_dir / "X_train_augmented.npy", X_balanced)
    np.save(output_dir / "y_train_augmented.npy", y_balanced)
    
    logger.info(f"Saved augmented data to:")
    logger.info(f"  {output_dir / 'X_train_augmented.npy'}")
    logger.info(f"  {output_dir / 'y_train_augmented.npy'}")
    
    logger.info("\n✅ Augmentation complete!")
    logger.info("\nNext step: Train with augmented data")
    logger.info("  python3 scripts/train.py --use-augmented")


if __name__ == "__main__":
    main()