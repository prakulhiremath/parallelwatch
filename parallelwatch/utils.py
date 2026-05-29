"""
Utility functions for ParallelWatch: normalization, metrics, data handling.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score


class StreamNormalizer:
    """Online normalization for streaming telemetry data."""
    
    def __init__(self, num_metrics: int, momentum: float = 0.95):
        self.num_metrics = num_metrics
        self.momentum = momentum
        self.mean = np.zeros(num_metrics)
        self.std = np.ones(num_metrics)
        self.M2 = np.zeros(num_metrics)
        self.n = 0
    
    def update(self, x: np.ndarray):
        """Welford's online algorithm for mean/variance."""
        assert x.shape[-1] == self.num_metrics
        
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        for sample in x:
            self.n += 1
            delta = sample - self.mean
            self.mean += delta / self.n
            delta2 = sample - self.mean
            self.M2 += delta * delta2
            
            if self.n > 1:
                self.std = np.sqrt(self.M2 / (self.n - 1))
            self.std = np.maximum(self.std, 1e-8)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize sample using running statistics."""
        return (x - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return x * (self.std + 1e-8) + self.mean


def compute_anomaly_metrics(
    anomaly_scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: Optional[float] = None
) -> dict:
    """
    Compute anomaly detection metrics.
    
    Args:
        anomaly_scores: Predicted anomaly scores [N]
        labels: Ground truth labels [N], 1 = anomaly, 0 = normal
        threshold: Detection threshold, if None use optimal via ROC
    
    Returns:
        Dictionary of metrics
    """
    scores_np = anomaly_scores.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    
    if len(np.unique(labels_np)) < 2:
        return {"warning": "Only one class in labels"}
    
    auroc = roc_auc_score(labels_np, scores_np)
    ap = average_precision_score(labels_np, scores_np)
    
    metrics = {
        "auroc": auroc,
        "ap": ap,
    }
    
    if threshold is not None:
        predictions = (scores_np >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels_np == 1))
        fp = np.sum((predictions == 1) & (labels_np == 0))
        tn = np.sum((predictions == 0) & (labels_np == 0))
        fn = np.sum((predictions == 0) & (labels_np == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    
    return metrics


def compute_cascade_metrics(
    cascade_scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: Optional[float] = None
) -> dict:
    """Compute cascade detection metrics."""
    return compute_anomaly_metrics(cascade_scores, labels, threshold)


def create_synthetic_cascade(
    num_metrics: int,
    sequence_length: int,
    cascade_start: int,
    cascade_end: int,
    cascade_indices: list,
    base_std: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic multivariate time series with cascading anomaly.
    
    Args:
        num_metrics: Number of metrics
        sequence_length: Length of sequence
        cascade_start: Cascade onset time
        cascade_end: Cascade end time
        cascade_indices: Metric indices involved in cascade
        base_std: Standard deviation of normal data
    
    Returns:
        data: [sequence_length, num_metrics]
        labels: [sequence_length, num_metrics]
    """
    data = np.random.normal(0, base_std, (sequence_length, num_metrics))
    labels = np.zeros((sequence_length, num_metrics), dtype=int)
    
    for t in range(cascade_start, cascade_end):
        amplitude = 0.5 * (t - cascade_start) / (cascade_end - cascade_start)
        for idx in cascade_indices:
            data[t, idx] += amplitude * np.random.normal(2.0, 0.5)
            labels[t, idx] = 1
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def batch_data(
    data: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    shuffle: bool = False
) -> list:
    """
    Create batches of sequences from flat data.
    
    Args:
        data: [num_samples, num_metrics]
        batch_size: Batch size
        sequence_length: Sequence length
        shuffle: Whether to shuffle batches
    
    Returns:
        List of batches [batch, sequence_length, num_metrics]
    """
    num_sequences = (data.shape[0] - sequence_length) // sequence_length + 1
    batches = []
    
    indices = np.arange(num_sequences)
    if shuffle:
        np.random.shuffle(indices)
    
    for batch_idx in range(0, len(indices), batch_size):
        batch_indices = indices[batch_idx:batch_idx + batch_size]
        batch_sequences = []
        
        for idx in batch_indices:
            start = idx * sequence_length
            end = start + sequence_length
            if end <= data.shape[0]:
                batch_sequences.append(data[start:end].unsqueeze(0))
        
        if batch_sequences:
            batch = torch.cat(batch_sequences, dim=0)
            batch = batch.unsqueeze(-1)
            batches.append(batch)
    
    return batches


def moving_average_filter(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Apply 1D moving average filtering."""
    if window_size < 2:
        return x
    
    padding = window_size // 2
    x_padded = F.pad(x.unsqueeze(1), (0, 0, padding, padding), mode='reflect')
    kernel = torch.ones(1, 1, window_size) / window_size
    x_filtered = F.conv1d(x_padded, kernel.to(x.device), padding=0).squeeze(1)
    
    return x_filtered


def exponential_smoothing(x: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Apply exponential smoothing."""
    smoothed = torch.zeros_like(x)
    smoothed[0] = x[0]
    
    for t in range(1, x.shape[0]):
        smoothed[t] = alpha * x[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed


def compute_state_stability(
    hidden_states: torch.Tensor,
    window_size: int = 10
) -> torch.Tensor:
    """
    Compute state trajectory stability (lower = more stable).
    
    Args:
        hidden_states: [batch, time, num_metrics, hidden_dim]
        window_size: Stability window
    
    Returns:
        stability_scores: [batch, time]
    """
    batch, time, num_metrics, hidden_dim = hidden_states.shape
    stability = torch.zeros(batch, time, device=hidden_states.device)
    
    for t in range(window_size, time):
        state_trajectory = hidden_states[:, t-window_size:t+1, :, :]
        velocity = torch.diff(state_trajectory, dim=1)
        acceleration = torch.diff(velocity, dim=1)
        
        stability[:, t] = torch.norm(acceleration, dim=(-2, -1)).mean(dim=1)
    
    return stability


def attention_to_correlation_matrix(
    attention_weights: torch.Tensor,
    num_samples: int = 100
) -> torch.Tensor:
    """
    Estimate metric correlation matrix from attention patterns.
    
    Args:
        attention_weights: [batch, time, num_metrics, num_metrics]
        num_samples: Number of time steps to average
    
    Returns:
        correlation_matrix: [num_metrics, num_metrics]
    """
    batch, time, num_metrics, _ = attention_weights.shape
    
    end_idx = min(num_samples, time)
    attention_slice = attention_weights[:, -end_idx:, :, :]
    
    correlation = attention_slice.mean(dim=(0, 1))
    
    return correlation
