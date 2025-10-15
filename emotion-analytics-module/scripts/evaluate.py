"""
Evaluation script for trained emotion classification model.
"""
import argparse
from pathlib import Path
import logging
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    EMOTION_LABELS,
    ACCURACY_THRESHOLD,
    PROCESSING_TIME_THRESHOLD,
)
from src.model import EmotionMLP
from src.pipeline import EmotionAnalyticsPipeline
from src.utils import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(data_dir: Path = PROCESSED_DATA_DIR):
    """Load test data."""
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    with open(data_dir / "metadata_test.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return X_test, y_test, metadata


def evaluate_model(
    model_path: Path,
    data_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = None
):
    """
    Evaluate trained model on test set.
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing test data
        output_dir: Directory to save evaluation results
    """
    if output_dir is None:
        output_dir = model_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EmotionMLP(
        input_size=checkpoint["input_size"],
        hidden_sizes=checkpoint["hidden_sizes"],
        output_size=checkpoint["output_size"],
        dropout=0.0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Load test data
    logger.info("Loading test data...")
    X_test, y_test, metadata = load_test_data(data_dir)
    
    # Load scaler
    scaler_path = model_path.parent / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(X_test)
    
    # Make predictions
    logger.info("Making predictions...")
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(y_test, y_pred)
    
    # Per-emotion metrics
    y_pred_binary = (y_pred >= 0.5).astype(int)
    y_test_binary = y_test.astype(int)  # Add this line!
    per_emotion_metrics = {}
    
    for i, emotion in enumerate(EMOTION_LABELS):
        emotion_metrics = {
            "accuracy": np.mean((y_test[:, i] == y_pred_binary[:, i]).astype(float)),
            "precision": np.sum((y_pred_binary[:, i] == 1) & (y_test_binary[:, i] == 1)) / 
                        (np.sum(y_pred_binary[:, i] == 1) + 1e-10),
            "recall": np.sum((y_pred_binary[:, i] == 1) & (y_test_binary[:, i] == 1)) / 
                     (np.sum(y_test[:, i] == 1) + 1e-10),
        }
        emotion_metrics["f1"] = 2 * (emotion_metrics["precision"] * emotion_metrics["recall"]) / \
                                (emotion_metrics["precision"] + emotion_metrics["recall"] + 1e-10)
        per_emotion_metrics[emotion] = emotion_metrics
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (Micro): {metrics['f1_micro']:.4f}")
    if metrics.get('roc_auc', 0.0) > 0:
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        logger.info("  ROC-AUC: Not available (insufficient class distribution)")
    
    logger.info(f"\nPer-Emotion Metrics:")
    for emotion, emetrics in per_emotion_metrics.items():
        logger.info(f"  {emotion.capitalize()}:")
        logger.info(f"    Accuracy: {emetrics['accuracy']:.4f}")
        logger.info(f"    Precision: {emetrics['precision']:.4f}")
        logger.info(f"    Recall: {emetrics['recall']:.4f}")
        logger.info(f"    F1: {emetrics['f1']:.4f}")
    
    # Check requirements
    logger.info(f"\nRequirements Check:")
    meets_accuracy = metrics['accuracy'] >= ACCURACY_THRESHOLD
    logger.info(f"  Accuracy >= {ACCURACY_THRESHOLD}: {'✓' if meets_accuracy else '✗'} "
                f"({metrics['accuracy']:.4f})")
    
    # Save metrics
    results = {
        "overall_metrics": metrics,
        "per_emotion_metrics": per_emotion_metrics,
        "requirements_met": {
            "accuracy": meets_accuracy,
            "threshold": ACCURACY_THRESHOLD,
        }
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSaved metrics to {output_dir / 'metrics.json'}")
    
    # Generate visualizations
    # generate_visualizations(y_test, y_pred, y_pred_binary, output_dir)
    generate_visualizations(y_test_binary, y_pred, y_pred_binary, output_dir)

    
    return results


def generate_visualizations(y_test, y_pred, y_pred_binary, output_dir):
    """Generate evaluation visualizations."""
    logger.info("Generating visualizations...")
    
    # 1. Confusion matrix for each emotion
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, emotion in enumerate(EMOTION_LABELS):
        cm = confusion_matrix(y_test[:, i], y_pred_binary[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{emotion.capitalize()} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300)
    plt.close()
    
    # 2. ROC curves
    from sklearn.metrics import roc_curve, auc
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, emotion in enumerate(EMOTION_LABELS):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{emotion.capitalize()} ROC Curve')
        axes[i].legend(loc="lower right")
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.close()
    
    # 3. Score distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, emotion in enumerate(EMOTION_LABELS):
        # Separate by true label
        scores_positive = y_pred[y_test[:, i] == 1, i]
        scores_negative = y_pred[y_test[:, i] == 0, i]
        
        axes[i].hist(scores_positive, bins=20, alpha=0.5, label='Positive', color='green')
        axes[i].hist(scores_negative, bins=20, alpha=0.5, label='Negative', color='red')
        axes[i].set_xlabel('Predicted Score')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{emotion.capitalize()} Score Distribution')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=300)
    plt.close()
    
    # 4. Per-emotion performance bar chart
    from sklearn.metrics import f1_score
    
    f1_scores = []
    for i in range(len(EMOTION_LABELS)):
        f1 = f1_score(y_test[:, i], y_pred_binary[:, i], zero_division=0)
        f1_scores.append(f1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(EMOTION_LABELS, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Emotion F1 Scores')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.75, color='r', linestyle='--', label='Target Threshold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "f1_scores.png", dpi=300)
    plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def benchmark_pipeline(
    model_path: Path,
    data_dir: Path = PROCESSED_DATA_DIR,
    num_samples: int = 50
):
    """
    Benchmark end-to-end pipeline performance.
    
    Args:
        model_path: Path to trained model
        data_dir: Directory containing test data
        num_samples: Number of samples to benchmark
    """
    logger.info("Benchmarking pipeline performance...")
    
    # Load metadata to get audio file paths
    with open(data_dir / "metadata_test.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Select samples
    audio_paths = [m["path"] for m in metadata[:num_samples]]
    
    # Initialize pipeline
    pipeline = EmotionAnalyticsPipeline(model_path=model_path)
    
    # Benchmark
    results = pipeline.validate_performance(audio_paths)
    
    logger.info("\nPipeline Performance:")
    logger.info(f"  Mean processing time: {results['mean_processing_time']:.4f}s")
    logger.info(f"  Median processing time: {results['median_processing_time']:.4f}s")
    logger.info(f"  Max processing time: {results['max_processing_time']:.4f}s")
    logger.info(f"  Meets latency requirement (<{PROCESSING_TIME_THRESHOLD}s): "
                f"{'✓' if results['meets_latency_requirement'] else '✗'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion classification model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=CHECKPOINTS_DIR / "best_model.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run pipeline performance benchmark"
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark_pipeline(
            model_path=args.model_path,
            data_dir=args.data_dir
        )


if __name__ == "__main__":
    main()