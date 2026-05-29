"""
ParallelWatch Engine: High-Performance Multivariate Temporal Correlation Detection
State-Space Model based cascade anomaly detection for infrastructure telemetry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Configuration for ParallelWatch Engine."""
    num_metrics: int
    hidden_dim: int
    num_attention_heads: int = 4
    dropout: float = 0.0
    eps: float = 1e-8
    device: str = "cpu"
    decay_rate_min: float = 0.90
    decay_rate_max: float = 0.99


class ParallelWatchEngine(nn.Module):
    """
    Parallel State-Space Model Engine for Multivariate Anomaly Detection.
    
    Processes M independent metric streams in parallel, maintaining per-metric
    hidden states. Computes cross-metric attention over states to detect cascading
    failures without expensive pairwise correlation computations.
    
    Args:
        config (EngineConfig): Engine configuration parameters.
    """
    
    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.num_metrics = config.num_metrics
        self.hidden_dim = config.hidden_dim
        self.eps = config.eps
        
        # Per-metric SSM parameters: A_i ⊙ h_{i,t-1} + B_i x_{i,t}
        # A_i: diagonal decay matrix, initialized log-uniform in (decay_rate_min, decay_rate_max)
        log_decay_min = np.log(config.decay_rate_min)
        log_decay_max = np.log(config.decay_rate_max)
        decay_init = np.exp(np.random.uniform(
            log_decay_min, log_decay_max, (config.num_metrics, config.hidden_dim)
        ))
        
        self.register_parameter(
            "decay_log",
            nn.Parameter(torch.tensor(np.log(decay_init), dtype=torch.float32))
        )
        
        # B_i: input projection per metric [num_metrics, hidden_dim]
        self.register_parameter(
            "input_proj",
            nn.Parameter(torch.randn(config.num_metrics, config.hidden_dim) * 0.01)
        )
        
        # Attention projection layers for cross-metric correlation
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Output projection for cascade score
        self.cascade_proj = nn.Linear(config.hidden_dim, 1)
        
        # Anomaly baseline: running statistics for adaptive thresholding
        self.register_buffer(
            "anomaly_baseline",
            torch.zeros(config.num_metrics, dtype=torch.float32)
        )
        self.register_buffer(
            "anomaly_variance",
            torch.ones(config.num_metrics, dtype=torch.float32)
        )
        self.register_buffer(
            "baseline_momentum",
            torch.tensor(0.95, dtype=torch.float32)
        )
        
        # Persistent streaming state: [num_metrics, hidden_dim]
        self.register_buffer(
            "hidden_state",
            torch.zeros(config.num_metrics, config.hidden_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "step_counter",
            torch.tensor(0, dtype=torch.int64)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all learnable parameters with stable distributions."""
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.cascade_proj.weight, gain=0.01)
        
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.cascade_proj.bias)
    
    def _get_decay_matrix(self) -> torch.Tensor:
        """
        Compute per-metric decay matrices from log-parameterized representation.
        
        Returns:
            torch.Tensor: [num_metrics, hidden_dim] diagonal decay values in (0, 1)
        """
        decay = torch.exp(self.decay_log).clamp(
            min=self.config.decay_rate_min,
            max=self.config.decay_rate_max
        )
        return decay
    
    def _ssm_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single SSM step: h_{i,t} = A_i ⊙ h_{i,t-1} + B_i x_{i,t}
        
        Args:
            x_t: Input at time t [batch, num_metrics, 1]
            h_prev: Previous hidden state [batch, num_metrics, hidden_dim]
        
        Returns:
            h_t: Updated hidden state [batch, num_metrics, hidden_dim]
            x_reconstructed: Reconstructed input [batch, num_metrics, 1]
        """
        batch_size = x_t.size(0)
        
        # Decay term: A_i ⊙ h_{i,t-1}
        decay = self._get_decay_matrix()  # [num_metrics, hidden_dim]
        decay = decay.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_metrics, hidden_dim]
        h_decay = decay * h_prev
        
        # Input projection: B_i x_{i,t}
        # x_t: [batch, num_metrics, 1] -> expand and project
        input_contrib = self.input_proj.unsqueeze(0) * x_t  # [batch, num_metrics, hidden_dim]
        
        # Combined update
        h_t = h_decay + input_contrib
        
        # Reconstruction: project back to input space (averaged over hidden)
        x_reconstructed = h_t.mean(dim=-1, keepdim=True)  # [batch, num_metrics, 1]
        
        return h_t, x_reconstructed
    
    def _compute_anomaly_scores(
        self,
        x_t: torch.Tensor,
        x_recon: torch.Tensor,
        h_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-metric anomaly scores from prediction error.
        
        Args:
            x_t: Actual input [batch, num_metrics, 1]
            x_recon: Reconstructed input [batch, num_metrics, 1]
            h_t: Hidden state [batch, num_metrics, hidden_dim]
        
        Returns:
            anomaly_scores: [batch, num_metrics]
        """
        # L2 prediction error
        pred_error = torch.abs(x_t.squeeze(-1) - x_recon.squeeze(-1))  # [batch, num_metrics]
        
        # State magnitude (captures dynamics)
        state_magnitude = torch.norm(h_t, dim=-1)  # [batch, num_metrics]
        
        # Combined anomaly signal
        anomaly = pred_error + 0.1 * state_magnitude
        
        return anomaly
    
    def _compute_cascade_score(
        self,
        h_t: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global cascade score from attention entropy and state variance.
        
        High entropy (uniform attention) = independent metrics = normal
        Low entropy (sharp peaks) = correlated metrics = cascade/failure
        
        Args:
            h_t: Hidden states [batch, num_metrics, hidden_dim]
            attention_weights: Attention matrix [batch, num_metrics, num_metrics]
        
        Returns:
            cascade_scores: [batch]
        """
        batch_size = h_t.size(0)
        
        # Attention entropy: -sum(p * log(p))
        # High entropy = normal; low entropy = anomaly
        attention_entropy = -(
            attention_weights * torch.log(attention_weights + self.eps)
        ).sum(dim=-1).mean(dim=-1)  # [batch]
        
        # Normalize entropy to [0, 1] scale
        max_entropy = np.log(self.num_metrics)
        normalized_entropy = attention_entropy / (max_entropy + self.eps)
        
        # Cascade score inverts entropy: high correlation -> low entropy -> high cascade score
        cascade_baseline = 0.5
        cascade_score = (cascade_baseline - normalized_entropy).clamp(min=0)
        
        # State variance across metrics (high variance = decorrelated = normal)
        state_variance = torch.var(h_t, dim=1).mean(dim=-1)  # [batch]
        state_variance_norm = torch.sigmoid(state_variance - 1.0)
        
        # Combined cascade score
        final_cascade = cascade_score + 0.3 * (1 - state_variance_norm)
        
        return final_cascade
    
    def forward(
        self,
        x: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over full sequence.
        
        Args:
            x: Input sequence [batch, time, num_metrics, 1]
            h_init: Initial hidden state [batch, num_metrics, hidden_dim], default zeros
            return_states: Whether to return hidden states at each timestep
        
        Returns:
            Dictionary with keys:
                - anomaly_scores: [batch, time, num_metrics]
                - cascade_scores: [batch, time]
                - attention_weights: [batch, time, num_metrics, num_metrics]
                - hidden_states: [batch, time, num_metrics, hidden_dim] (if return_states=True)
        """
        batch_size, time_steps, num_metrics, _ = x.shape
        assert num_metrics == self.num_metrics, \
            f"Input has {num_metrics} metrics, expected {self.num_metrics}"
        
        device = x.device
        
        # Initialize hidden state
        if h_init is None:
            h = torch.zeros(
                batch_size, self.num_metrics, self.hidden_dim,
                dtype=x.dtype, device=device
            )
        else:
            h = h_init.to(device)
        
        # Storage for outputs
        anomaly_scores_list = []
        cascade_scores_list = []
        attention_weights_list = []
        hidden_states_list = [] if return_states else None
        
        # Process sequence
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # [batch, num_metrics, 1]
            
            # SSM step
            h, x_recon = self._ssm_step(x_t, h)
            
            # Anomaly scores
            anomaly = self._compute_anomaly_scores(x_t, x_recon, h)
            anomaly_scores_list.append(anomaly)
            
            # Cross-metric attention for cascade detection
            # Project states to query/key/value space
            h_flat = h.view(batch_size * self.num_metrics, self.hidden_dim)  # [B*M, d]
            
            q = self.query_proj(h_flat).view(batch_size, self.num_metrics, self.hidden_dim)
            k = self.key_proj(h_flat).view(batch_size, self.num_metrics, self.hidden_dim)
            v = self.value_proj(h_flat).view(batch_size, self.num_metrics, self.hidden_dim)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
            attention_weights = F.softmax(scores, dim=-1)
            
            attention_weights_list.append(attention_weights)
            
            # Cascade score
            cascade = self._compute_cascade_score(h, attention_weights)
            cascade_scores_list.append(cascade)
            
            if return_states:
                hidden_states_list.append(h)
        
        # Stack outputs
        anomaly_scores = torch.stack(anomaly_scores_list, dim=1)  # [batch, time, num_metrics]
        cascade_scores = torch.stack(cascade_scores_list, dim=1)  # [batch, time]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch, time, M, M]
        
        output = {
            "anomaly_scores": anomaly_scores,
            "cascade_scores": cascade_scores,
            "attention_weights": attention_weights,
        }
        
        if return_states:
            hidden_states = torch.stack(hidden_states_list, dim=1)  # [batch, time, num_metrics, hidden_dim]
            output["hidden_states"] = hidden_states
        
        return output
    
    def step(
        self,
        x_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single-step streaming mode: process one timestamp and update persistent state.
        
        Args:
            x_t: Input at current timestamp [num_metrics] or [batch, num_metrics]
        
        Returns:
            Dictionary with:
                - anomaly_scores: [num_metrics] or [batch, num_metrics]
                - cascade_score: scalar or [batch]
                - attention_weights: [num_metrics, num_metrics] or [batch, num_metrics, num_metrics]
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0).unsqueeze(-1)  # [1, num_metrics, 1]
            squeeze_output = True
        elif x_t.dim() == 2:
            x_t = x_t.unsqueeze(-1)  # [batch, num_metrics, 1]
            squeeze_output = False
        else:
            squeeze_output = False
        
        batch_size = x_t.size(0)
        device = x_t.device
        
        # Expand persistent state if batch size changed
        if self.hidden_state.size(0) != batch_size:
            self.hidden_state = self.hidden_state[:1].expand(
                batch_size, -1, -1
            ).clone().to(device)
        else:
            self.hidden_state = self.hidden_state.to(device)
        
        # SSM step
        h_new, x_recon = self._ssm_step(x_t, self.hidden_state)
        self.hidden_state = h_new.detach()
        
        # Anomaly scores
        anomaly = self._compute_anomaly_scores(x_t, x_recon, h_new)
        
        # Update anomaly baseline with exponential moving average
        with torch.no_grad():
            momentum = self.baseline_momentum
            self.anomaly_baseline = (
                momentum * self.anomaly_baseline +
                (1 - momentum) * anomaly.mean(dim=0)
            )
            self.anomaly_variance = (
                momentum * self.anomaly_variance +
                (1 - momentum) * anomaly.var(dim=0)
            )
        
        # Normalized anomaly score
        anomaly_normalized = (
            (anomaly - self.anomaly_baseline.unsqueeze(0)) /
            (self.anomaly_variance.unsqueeze(0).sqrt() + self.eps)
        )
        
        # Cascade detection via attention
        h_flat = h_new.view(batch_size * self.num_metrics, self.hidden_dim)
        q = self.query_proj(h_flat).view(batch_size, self.num_metrics, self.hidden_dim)
        k = self.key_proj(h_flat).view(batch_size, self.num_metrics, self.hidden_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        cascade = self._compute_cascade_score(h_new, attention_weights)
        
        # Normalize cascade score
        cascade_normalized = torch.sigmoid(cascade - 0.5)
        
        output = {
            "anomaly_scores": anomaly_normalized.squeeze(0) if squeeze_output else anomaly_normalized,
            "cascade_score": cascade_normalized.squeeze(0) if squeeze_output else cascade_normalized,
            "attention_weights": attention_weights.squeeze(0) if squeeze_output else attention_weights,
        }
        
        self.step_counter += 1
        
        return output
    
    def reset_state(self):
        """Reset persistent hidden state and step counter for new stream."""
        self.hidden_state.zero_()
        self.step_counter.zero_()
        self.anomaly_baseline.zero_()
        self.anomaly_variance.fill_(1.0)
    
    def get_state(self) -> torch.Tensor:
        """Get current hidden state."""
        return self.hidden_state.clone()
    
    def set_state(self, state: torch.Tensor):
        """Set hidden state (for resuming from checkpoint)."""
        assert state.shape == self.hidden_state.shape, \
            f"State shape mismatch: got {state.shape}, expected {self.hidden_state.shape}"
        self.hidden_state = state.clone().to(self.hidden_state.device)
