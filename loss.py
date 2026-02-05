"""
Training and evaluation utilities for cross-modal models.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


def get_target_train_mean(base):
    """Extract target modality training mean from base dataset."""
    if base.target == "SC":
        return base.sc_train_avg
    elif base.target == "FC":
        return base.fc_train_avg
    else:
        raise ValueError(f"Unknown target modality: {base.target}")


# =============================================================================
# Loss Functions
# =============================================================================

class MSELoss(nn.Module):
    """Standard MSE loss."""
    
    def __init__(self):
        super().__init__()
        self.name = "mse"
    
    def forward(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)


class DemeanedMSELoss(nn.Module):
    """
    MSE loss on demeaned predictions and targets.
    Measures how well the model captures deviations from the training mean.
    
    Loss = MSE(y_pred - μ_train, y_true - μ_train)
    
    Args:
        target_train_mean: (d,) numpy array or tensor, training set mean for target modality
    """
    
    def __init__(self, target_train_mean):
        super().__init__()
        self.name = "demeaned_mse"
        if isinstance(target_train_mean, np.ndarray):
            target_train_mean = torch.tensor(target_train_mean, dtype=torch.float32)
        self.register_buffer('target_mean', target_train_mean)
    
    def forward(self, y_pred, y_true):
        y_pred_demeaned = y_pred - self.target_mean
        y_true_demeaned = y_true - self.target_mean
        return F.mse_loss(y_pred_demeaned, y_true_demeaned)


class WeightedMSELoss(nn.Module):
    """
    Weighted combination of standard MSE and demeaned MSE.
    
    Loss = α * MSE(y_pred, y_true) + (1-α) * MSE(y_pred - μ, y_true - μ)
    
    Both terms are normalized by their initial scale (computed from first batch)
    so they contribute equally when α=0.5.
    
    Args:
        target_train_mean: (d,) numpy array or tensor, training set mean for target modality
        alpha: weight for standard MSE (default 0.5). Higher = more weight on absolute accuracy.
    """
    
    def __init__(self, target_train_mean, alpha=0.5):
        super().__init__()
        self.name = "weighted_mse"
        self.alpha = alpha
        if isinstance(target_train_mean, np.ndarray):
            target_train_mean = torch.tensor(target_train_mean, dtype=torch.float32)
        self.register_buffer('target_mean', target_train_mean)
        
        # Normalization factors (estimated from first batch, then frozen)
        self.register_buffer('mse_scale', torch.tensor(1.0))
        self.register_buffer('demeaned_scale', torch.tensor(1.0))
        self._initialized = False
    
    def forward(self, y_pred, y_true):
        # Compute both losses
        mse_loss = F.mse_loss(y_pred, y_true)
        
        y_pred_demeaned = y_pred - self.target_mean
        y_true_demeaned = y_true - self.target_mean
        demeaned_loss = F.mse_loss(y_pred_demeaned, y_true_demeaned)
        
        # Initialize normalization scales from first batch
        if not self._initialized and self.training:
            with torch.no_grad(): # this dynamically standardizes to the first loss computation giving a rough estimate of the error magnitude of each
                self.mse_scale = mse_loss.clone()
                self.demeaned_scale = demeaned_loss.clone()
                self._initialized = True
        
        # Normalize both losses to similar scale
        mse_normalized = mse_loss / (self.mse_scale + 1e-8)
        demeaned_normalized = demeaned_loss / (self.demeaned_scale + 1e-8)
        
        # Weighted combination
        return self.alpha * mse_normalized + (1 - self.alpha) * demeaned_normalized


def create_loss_fn(loss_type, base=None, alpha=0.5):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: one of 'mse', 'demeaned_mse', 'weighted_mse'
        base: Dataset base object (required for demeaned losses)
        alpha: weight for weighted_mse (default 0.5)
    
    Returns:
        Loss function module
    """
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "demeaned_mse":
        if base is None:
            raise ValueError("base is required for demeaned_mse loss")
        target_mean = get_target_train_mean(base)
        return DemeanedMSELoss(target_mean)
    elif loss_type == "weighted_mse":
        if base is None:
            raise ValueError("base is required for weighted_mse loss")
        target_mean = get_target_train_mean(base)
        return WeightedMSELoss(target_mean, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from 'mse', 'demeaned_mse', 'weighted_mse'")


def compute_pearson_r(y_pred, y_true):
    """
    Compute mean Pearson correlation across samples.
    Each sample's prediction is correlated with its target across features.
    
    Args:
        y_pred: (batch, d) predicted values
        y_true: (batch, d) true values
    Returns:
        mean_r: scalar, mean Pearson r across batch
    """
    # Center each sample
    y_pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true_centered = y_true - y_true.mean(dim=1, keepdim=True)
    
    # Compute correlation per sample
    numerator = (y_pred_centered * y_true_centered).sum(dim=1)
    denom_pred = torch.sqrt((y_pred_centered ** 2).sum(dim=1))
    denom_true = torch.sqrt((y_true_centered ** 2).sum(dim=1))
    
    r = numerator / (denom_pred * denom_true + 1e-10)
    return r.mean().item()


def compute_demeaned_pearson_r(y_pred, y_true, target_train_mean):
    """
    Compute demeaned Pearson correlation across samples.
    Correlation is computed between (y_pred - μ_train) and (y_true - μ_train).
    This measures how well we capture individual deviations from the population mean.
    
    Args:
        y_pred: (batch, d) predicted values
        y_true: (batch, d) true values
        target_train_mean: (d,) training set mean for target modality
    Returns:
        mean_r: scalar, mean demeaned Pearson r across batch
    """
    # Demean by training set mean
    y_pred_demeaned = y_pred - target_train_mean
    y_true_demeaned = y_true - target_train_mean
    
    # Compute correlation per sample
    numerator = (y_pred_demeaned * y_true_demeaned).sum(dim=1)
    denom_pred = torch.sqrt((y_pred_demeaned ** 2).sum(dim=1))
    denom_true = torch.sqrt((y_true_demeaned ** 2).sum(dim=1))
    
    r = numerator / (denom_pred * denom_true + 1e-10)
    return r.mean().item()


def evaluate_model(model, data_loader, target_train_mean, device):
    """
    Evaluate model on a data loader.
    
    Args:
        model: the model to evaluate
        data_loader: DataLoader yielding batches with 'x' and 'y' keys
        target_train_mean: (d,) training mean for target modality (for demeaned corr)
        device: torch device
        
    Returns:
        dict with 'mse', 'pearson_r', 'demeaned_r'
    """
    model.eval()
    total_mse = 0.0
    total_pearson_r = 0.0
    total_demeaned_r = 0.0
    n_batches = 0
    
    target_mean_tensor = torch.tensor(target_train_mean, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            y_pred = model(x)
            
            mse = F.mse_loss(y_pred, y).item()
            total_mse += mse
            
            pearson_r = compute_pearson_r(y_pred, y)
            total_pearson_r += pearson_r
            
            demeaned_r = compute_demeaned_pearson_r(y_pred, y, target_mean_tensor)
            total_demeaned_r += demeaned_r
            
            n_batches += 1
    
    return {
        'mse': total_mse / n_batches,
        'pearson_r': total_pearson_r / n_batches,
        'demeaned_r': total_demeaned_r / n_batches,
    }


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0.0, 0.0


class TrainingLogger:
    """Logger for tracking and plotting training metrics.
    
    Text output accumulates (persists), figure updates in place.
    """
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_mse': [],
            'train_pearson_r': [],
            'train_demeaned_r': [],
            'val_mse': [],
            'val_pearson_r': [],
            'val_demeaned_r': [],
            'epoch_time': [],
            'gpu_allocated_mb': [],
            'gpu_reserved_mb': [],
        }
        self.fig = None
        self.axes = None
        self.display_handle = None
        
    def log(self, epoch, train_metrics, val_metrics, epoch_time, gpu_alloc, gpu_reserved):
        """Log metrics for an epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_mse'].append(train_metrics['mse'])
        self.history['train_pearson_r'].append(train_metrics['pearson_r'])
        self.history['train_demeaned_r'].append(train_metrics['demeaned_r'])
        self.history['val_mse'].append(val_metrics['mse'])
        self.history['val_pearson_r'].append(val_metrics['pearson_r'])
        self.history['val_demeaned_r'].append(val_metrics['demeaned_r'])
        self.history['epoch_time'].append(epoch_time)
        self.history['gpu_allocated_mb'].append(gpu_alloc)
        self.history['gpu_reserved_mb'].append(gpu_reserved)
        
    def print_metrics(self, epoch):
        """Print metrics for current epoch (accumulates in output)."""
        idx = self.history['epoch'].index(epoch)
        print(f"Epoch {epoch:4d} | "
              f"Train MSE: {self.history['train_mse'][idx]:.6f}, r: {self.history['train_pearson_r'][idx]:.4f}, r_dm: {self.history['train_demeaned_r'][idx]:.4f} | "
              f"Val MSE: {self.history['val_mse'][idx]:.6f}, r: {self.history['val_pearson_r'][idx]:.4f}, r_dm: {self.history['val_demeaned_r'][idx]:.4f} | "
              f"{self.history['epoch_time'][idx]:.2f}s | "
              f"GPU: {self.history['gpu_allocated_mb'][idx]:.0f}MB")
        
    def plot(self):
        """Update the training curves plot in place."""
        if len(self.history['epoch']) == 0:
            return
        
        # Create figure on first call
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
            plt.tight_layout()
            # Create display handle for in-place updates
            self.display_handle = display(self.fig, display_id=True)
        
        # Clear and redraw axes
        for ax in self.axes:
            ax.clear()
        
        epochs = self.history['epoch']
        
        # Plot 1: MSE Loss
        self.axes[0].plot(epochs, self.history['train_mse'], 'b-o', label='Train', markersize=4)
        self.axes[0].plot(epochs, self.history['val_mse'], 'r-o', label='Val', markersize=4)
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('MSE Loss')
        self.axes[0].set_title('MSE Loss')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Pearson r
        self.axes[1].plot(epochs, self.history['train_pearson_r'], 'b-o', label='Train', markersize=4)
        self.axes[1].plot(epochs, self.history['val_pearson_r'], 'r-o', label='Val', markersize=4)
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Pearson r')
        self.axes[1].set_title('Pearson Correlation')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Demeaned Pearson r
        self.axes[2].plot(epochs, self.history['train_demeaned_r'], 'b-o', label='Train', markersize=4)
        self.axes[2].plot(epochs, self.history['val_demeaned_r'], 'r-o', label='Val', markersize=4)
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('Demeaned Pearson r')
        self.axes[2].set_title('Demeaned Correlation')
        self.axes[2].legend()
        self.axes[2].grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Update the existing display (in place)
        self.display_handle.update(self.fig)
    
    def close(self):
        """Close the figure to free resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self.display_handle = None


def train_model(
    model,
    train_loader,
    val_loader,
    base,
    log_every=5,
    device=None,
):
    """
    Train a cross-modal model.
    
    Args:
        model: nn.Module to train. Uses model.lr, model.epochs, model.loss_fn, model.get_reg_loss().
        train_loader: DataLoader for training data (batches with 'x', 'y' keys)
        val_loader: DataLoader for validation data
        base: Dataset base object (e.g., HCP_Base) with target modality info
        log_every: log metrics and update plot every N epochs
        device: torch device
        
    Returns:
        logger: TrainingLogger with full history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_train_mean = get_target_train_mean(base)
    
    model = model.to(device)
    
    # Get training params from model
    lr = getattr(model, 'lr', 1e-4)
    num_epochs = getattr(model, 'epochs', 100)
    
    # Get loss function from model or default to MSE
    if hasattr(model, 'loss_fn') and model.loss_fn is not None:
        loss_fn = model.loss_fn
        loss_fn = loss_fn.to(device)
        loss_name = getattr(loss_fn, 'name', type(loss_fn).__name__)
    else:
        loss_fn = MSELoss().to(device)
        loss_name = "mse"
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger = TrainingLogger()
    
    reg_info = f"L1={model.l1_reg}, L2={model.l2_reg}" if hasattr(model, 'l1_reg') else "none"
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Loss: {loss_name} | Reg: {reg_info}")
    print(f"Epochs: {num_epochs}, LR: {lr}")
    print(f"Logging every {log_every} epochs\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # === Training ===
        model.train()
        loss_fn.train()
        train_loss_accum = 0.0
        n_train_batches = 0
        
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            # Add regularization if model has get_reg_loss method
            if hasattr(model, 'get_reg_loss'):
                loss = loss + model.get_reg_loss()
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            n_train_batches += 1
        
        epoch_time = time.time() - epoch_start
        
        # === Logging every log_every epochs ===
        if epoch % log_every == 0 or epoch == 1:
            train_metrics = evaluate_model(model, train_loader, target_train_mean, device)
            val_metrics = evaluate_model(model, val_loader, target_train_mean, device)
            gpu_alloc, gpu_reserved = get_gpu_memory_usage()
            logger.log(epoch, train_metrics, val_metrics, epoch_time, gpu_alloc, gpu_reserved)
            logger.print_metrics(epoch)
            logger.plot()
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    return logger


def train_model_early_stopping(
    model,
    train_loader,
    val_loader,
    base,
    log_every=5,
    patience=20,
    device=None,
):
    """
    Train with early stopping based on validation MSE.
    
    Args:
        model: nn.Module to train. Uses model.lr, model.epochs, model.loss_fn, model.get_reg_loss().
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        base: Dataset base object (e.g., HCP_Base) with target modality info
        log_every: log metrics every N epochs
        patience: stop if val MSE doesn't improve for this many epochs
        device: torch device
        
    Returns:
        logger: TrainingLogger with full history
        best_state_dict: state dict of best model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_train_mean = get_target_train_mean(base)
    
    model = model.to(device)
    
    # Get training params from model
    lr = getattr(model, 'lr', 1e-4)
    num_epochs = getattr(model, 'epochs', 100)
    
    # Get loss function from model or default to MSE
    if hasattr(model, 'loss_fn') and model.loss_fn is not None:
        loss_fn = model.loss_fn
        loss_fn = loss_fn.to(device)
        loss_name = getattr(loss_fn, 'name', type(loss_fn).__name__)
    else:
        loss_fn = MSELoss().to(device)
        loss_name = "mse"
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger = TrainingLogger()
    best_val_mse = float('inf')
    best_state_dict = None
    epochs_without_improvement = 0
    
    reg_info = f"L1={model.l1_reg}, L2={model.l2_reg}" if hasattr(model, 'l1_reg') else "none"
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Loss: {loss_name} | Reg: {reg_info}")
    print(f"Epochs: {num_epochs}, LR: {lr}, Patience: {patience}")
    print(f"Logging every {log_every} epochs\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # === Training ===
        model.train()
        loss_fn.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            # Add regularization if model has get_reg_loss method
            if hasattr(model, 'get_reg_loss'):
                loss = loss + model.get_reg_loss()
            
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - epoch_start
        
        # === Logging every log_every epochs ===
        if epoch % log_every == 0 or epoch == 1:
            train_metrics = evaluate_model(model, train_loader, target_train_mean, device)
            val_metrics = evaluate_model(model, val_loader, target_train_mean, device)
            gpu_alloc, gpu_reserved = get_gpu_memory_usage()
            logger.log(epoch, train_metrics, val_metrics, epoch_time, gpu_alloc, gpu_reserved)
            logger.print_metrics(epoch)
            logger.plot()
            
            # Early stopping check
            if val_metrics['mse'] < best_val_mse:
                best_val_mse = val_metrics['mse']
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                print(f"  ★ New best val MSE: {best_val_mse:.6f}")
            else:
                epochs_without_improvement += log_every
                
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    print("\n" + "="*80)
    print(f"Training complete! Best val MSE: {best_val_mse:.6f}")
    print("="*80)
    
    return logger, best_state_dict