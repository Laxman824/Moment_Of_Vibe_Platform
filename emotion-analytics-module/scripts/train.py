
"""
Training script for emotion classification model.
FIXED VERSION - Handles model output correctly
"""
# ===== FIX PATH ISSUE =====
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# ==========================

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import pickle

from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    ACCURACY_THRESHOLD,
)
from src.model import EmotionMLP
from src.utils import calculate_metrics, set_seed

import argparse
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    ACCURACY_THRESHOLD,
)
from src.model import EmotionMLP
from src.utils import calculate_metrics, set_seed

logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=15, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.mode = mode
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # min mode
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            else:
                self.best_score = score
                self.counter = 0


def compute_class_weights(y_train):
    """
    Compute class weights for imbalanced dataset.
    
    Returns:
        torch.Tensor: Weights for each class
    """
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    
    # Weight = neg_count / pos_count (balance classes)
    # Cap maximum weight to prevent extreme values
    weights = neg_counts / (pos_counts + 1e-6)
    weights = np.clip(weights, 0.5, 5.0)  # More conservative clipping
    
    logger.info("\n" + "="*60)
    logger.info("CLASS IMBALANCE ANALYSIS:")
    logger.info("="*60)
    for i, (pos, neg, weight) in enumerate(zip(pos_counts, neg_counts, weights)):
        logger.info(
            f"  Emotion {i}: "
            f"positive={pos:.0f} ({pos/len(y_train)*100:.1f}%), "
            f"negative={neg:.0f} ({neg/len(y_train)*100:.1f}%), "
            f"weight={weight:.2f}"
        )
    logger.info("="*60 + "\n")
    
    return torch.FloatTensor(weights)


def load_data(data_dir: Path = PROCESSED_DATA_DIR):
    """Load processed training data."""
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    
    return X_train, y_train, X_val, y_val


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X = batch_X.to(device).float()
        batch_y = batch_y.to(device).float()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        
        # Model outputs probabilities (has sigmoid)
        # Use BCELoss which expects probabilities in [0, 1]
        loss = criterion(outputs, batch_y)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss in batch {batch_idx}: {loss.item()}")
            logger.error(f"Outputs: min={outputs.min().item():.3f}, max={outputs.max().item():.3f}")
            logger.error(f"Targets: min={batch_y.min().item():.3f}, max={batch_y.max().item():.3f}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device).float()
            batch_y = batch_y.to(device).float()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


def create_weighted_bce_loss(pos_weights):
    """
    Create weighted BCE loss that handles probabilities correctly.
    """
    class WeightedBCELoss(nn.Module):
        def __init__(self, pos_weights):
            super().__init__()
            self.pos_weights = pos_weights
        
        def forward(self, outputs, targets):
            # outputs are probabilities [0, 1]
            # Clamp to avoid log(0)
            outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)
            
            # Manual BCE with weights
            loss = -(
                self.pos_weights * targets * torch.log(outputs) +
                (1 - targets) * torch.log(1 - outputs)
            )
            return loss.mean()
    
    return WeightedBCELoss(pos_weights)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    save_path: Path,
    scaler: StandardScaler,
    class_weights: torch.Tensor = None,
    early_stopping_patience: int = 15
):
    """
    Train emotion classification model.
    """
    
    # Use correct loss function
    if class_weights is not None:
        logger.info("Using weighted BCE loss to handle class imbalance")
        criterion = create_weighted_bce_loss(class_weights.to(device))
    else:
        logger.info("Using standard BCE loss")
        criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAINING_CONFIG["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize accuracy
        patience=7,
        factor=0.5,
        min_lr=1e-6,
        verbose=True
    )
    
    # Early stopping based on accuracy
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    
    best_val_acc = 0
    best_val_f1 = 0
    logger.info(f"\nStarting training for {num_epochs} epochs")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_metrics["accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} - LR: {current_lr:.2e} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}"
            )
        
        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        
        # Save best model based on accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_f1 = val_metrics["f1_macro"]
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler": scaler,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_macro"],
                "input_size": model.input_size,
                "hidden_sizes": model.hidden_sizes,
                "output_size": model.output_size,
            }, save_path)
            
            logger.info(f"✅ Saved best model - Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f}")
        
        # Early stopping based on accuracy
        early_stopping(val_metrics["accuracy"])
        if early_stopping.should_stop:
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"{'='*60}\n")
    
    if best_val_acc < ACCURACY_THRESHOLD:
        logger.warning(
            f"⚠️ Best accuracy {best_val_acc:.4f} is below threshold {ACCURACY_THRESHOLD}\n"
            "Recommendations:\n"
            "  1. Run data augmentation: python3 scripts/augment_data.py\n"
            "  2. Collect more balanced data\n"
            "  3. Try larger model: --hidden-sizes 512 256 128\n"
            "  4. Increase epochs: --epochs 200"
        )
    else:
        logger.info(f"✅ Model meets accuracy threshold ({ACCURACY_THRESHOLD})!")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training"
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        help="Use class weights to handle imbalance"
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs='+',
        type=int,
        default=[256, 128, 64],
        help="Hidden layer sizes"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate"
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("EMOTION CLASSIFICATION MODEL TRAINING")
    logger.info("="*60 + "\n")
    
    # Set seed
    set_seed(TRAINING_CONFIG["random_seed"])
    
    # Load data
    logger.info("Loading data...")
    X_train, y_train, X_val, y_val = load_data(args.data_dir)
    
    logger.info(f"  Train set: {X_train.shape[0]} samples")
    logger.info(f"  Validation set: {X_val.shape[0]} samples")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Emotions: {y_train.shape[1]}\n")
    
    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(y_train)
    
    # Normalize features
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    logger.info(f"  Feature mean: {X_train.mean():.3f}")
    logger.info(f"  Feature std: {X_train.std():.3f}\n")
    
    # Save scaler
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {args.output_dir / 'scaler.pkl'}\n")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize model
    logger.info("Initializing model...")
    logger.info(f"  Architecture: {X_train.shape[1]} -> {args.hidden_sizes} -> {y_train.shape[1]}")
    logger.info(f"  Dropout: {args.dropout}")
    
    model = EmotionMLP(
        input_size=X_train.shape[1],
        hidden_sizes=args.hidden_sizes,
        output_size=y_train.shape[1],
        dropout=args.dropout
    )
    model = model.to(args.device)
    
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    save_path = args.output_dir / "best_model.pth"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_path=save_path,
        scaler=scaler,
        class_weights=class_weights,
        early_stopping_patience=20
    )
    
    # Save history
    with open(args.output_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history to {args.output_dir / 'training_history.pkl'}")
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].axhline(y=ACCURACY_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({ACCURACY_THRESHOLD})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(args.output_dir / 'training_curves.png', dpi=150)
        logger.info(f"Saved training curves to {args.output_dir / 'training_curves.png'}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
    
    logger.info("\n✅ Training complete!\n")


if __name__ == "__main__":
    main()
