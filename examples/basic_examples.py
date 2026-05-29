"""
Example: Streaming Anomaly Detection with ParallelWatch
Demonstrates real-time anomaly detection on infrastructure telemetry.
"""

import torch
import numpy as np
from parallelwatch import ParallelWatchEngine, EngineConfig
from parallelwatch.utils import StreamNormalizer, create_synthetic_cascade


def example_basic_streaming():
    """Basic streaming mode with synthetic telemetry."""
    print("=" * 60)
    print("ParallelWatch: Basic Streaming Anomaly Detection")
    print("=" * 60)
    
    config = EngineConfig(
        num_metrics=20,
        hidden_dim=128,
        num_attention_heads=4
    )
    engine = ParallelWatchEngine(config)
    engine.reset_state()
    
    normalizer = StreamNormalizer(num_metrics=20, momentum=0.99)
    
    anomaly_log = []
    cascade_log = []
    
    for step in range(500):
        if step < 100:
            x_t = np.random.normal(0, 0.1, 20)
        elif step < 200:
            x_t = np.random.normal(0, 0.1, 20)
            x_t[0:5] += 2.0
        else:
            x_t = np.random.normal(0, 0.1, 20)
        
        normalizer.update(x_t)
        x_normalized = normalizer.normalize(x_t)
        x_tensor = torch.FloatTensor(x_normalized)
        
        output = engine.step(x_tensor)
        
        anomaly_scores = output["anomaly_scores"]
        cascade_score = output["cascade_score"].item()
        
        anomaly_log.append(anomaly_scores.mean().item())
        cascade_log.append(cascade_score)
        
        if step % 100 == 0:
            print(f"Step {step:3d} | Anomaly: {anomaly_scores.mean():.4f} | Cascade: {cascade_score:.4f}")
    
    print(f"\nMean anomaly score (normal period): {np.mean(anomaly_log[0:50]):.4f}")
    print(f"Mean anomaly score (cascade period): {np.mean(anomaly_log[100:150]):.4f}")
    print(f"Detection gain: {np.mean(anomaly_log[100:150]) / (np.mean(anomaly_log[0:50]) + 1e-8):.2f}x")


def example_batch_inference():
    """Batch inference on pre-recorded sequences."""
    print("\n" + "=" * 60)
    print("ParallelWatch: Batch Inference")
    print("=" * 60)
    
    config = EngineConfig(
        num_metrics=30,
        hidden_dim=256,
        num_attention_heads=8
    )
    engine = ParallelWatchEngine(config)
    
    batch_size = 8
    time_steps = 64
    num_metrics = 30
    
    x = torch.randn(batch_size, time_steps, num_metrics, 1) * 0.1
    x[2:4, 20:40, 5:10, :] += 1.5
    
    output = engine.forward(x, return_states=True)
    
    anomaly_scores = output["anomaly_scores"]
    cascade_scores = output["cascade_scores"]
    attention_weights = output["attention_weights"]
    hidden_states = output["hidden_states"]
    
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Cascade scores shape: {cascade_scores.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    print(f"\nBatch 0 mean anomaly: {anomaly_scores[0].mean():.4f}")
    print(f"Batch 2 mean anomaly: {anomaly_scores[2].mean():.4f} (has injected anomaly)")
    print(f"Batch 3 mean anomaly: {anomaly_scores[3].mean():.4f} (has injected anomaly)")
    
    print(f"\nCascade detection:")
    print(f"Batch 0 max cascade: {cascade_scores[0].max():.4f}")
    print(f"Batch 2 max cascade: {cascade_scores[2].max():.4f}")


def example_cascade_detection():
    """Detect synthetic infrastructure cascade."""
    print("\n" + "=" * 60)
    print("ParallelWatch: Cascade Detection")
    print("=" * 60)
    
    data, labels = create_synthetic_cascade(
        num_metrics=50,
        sequence_length=200,
        cascade_start=80,
        cascade_end=140,
        cascade_indices=[0, 1, 2, 3, 4],
        base_std=0.15
    )
    
    config = EngineConfig(
        num_metrics=50,
        hidden_dim=256,
        num_attention_heads=8
    )
    engine = ParallelWatchEngine(config)
    engine.reset_state()
    
    cascade_scores = []
    
    for t in range(data.shape[0]):
        x_t = data[t]
        output = engine.step(x_t)
        cascade_scores.append(output["cascade_score"].item())
    
    cascade_scores = np.array(cascade_scores)
    
    print(f"Normal period (0-79) cascade: {cascade_scores[0:80].mean():.4f}")
    print(f"Cascade period (80-140) cascade: {cascade_scores[80:140].mean():.4f}")
    print(f"Recovery period (140-200) cascade: {cascade_scores[140:200].mean():.4f}")
    print(f"\nCascade amplification: {cascade_scores[80:140].mean() / cascade_scores[0:80].mean():.2f}x")


def example_attention_visualization():
    """Visualize learned attention patterns over metrics."""
    print("\n" + "=" * 60)
    print("ParallelWatch: Attention Analysis")
    print("=" * 60)
    
    config = EngineConfig(
        num_metrics=10,
        hidden_dim=128,
        num_attention_heads=4
    )
    engine = ParallelWatchEngine(config)
    engine.reset_state()
    
    x = torch.randn(32, 10, 1) * 0.1
    
    attention_matrices = []
    for t in range(32):
        output = engine.step(x[t])
        attention = output["attention_weights"]
        attention_matrices.append(attention.cpu().numpy())
    
    attention_over_time = np.array(attention_matrices)
    print(f"Attention shape over time: {attention_over_time.shape}")
    
    mean_attention = attention_over_time.mean(axis=0)
    print(f"\nMean attention matrix (10x10):")
    print(f"Diagonal elements (self-attention): {np.diag(mean_attention)[:5]}")
    print(f"Max off-diagonal: {(mean_attention - np.eye(10)).max():.4f}")


def example_state_management():
    """Demonstrate state save/load for checkpointing."""
    print("\n" + "=" * 60)
    print("ParallelWatch: State Management")
    print("=" * 60)
    
    config = EngineConfig(num_metrics=15, hidden_dim=128)
    engine = ParallelWatchEngine(config)
    
    engine.step(torch.randn(15))
    engine.step(torch.randn(15))
    state_checkpoint = engine.get_state()
    
    engine.step(torch.randn(15))
    engine.step(torch.randn(15))
    
    engine.set_state(state_checkpoint)
    state_restored = engine.get_state()
    
    print(f"State checkpoint shape: {state_checkpoint.shape}")
    print(f"State restored matches checkpoint: {torch.allclose(state_restored, state_checkpoint)}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    example_basic_streaming()
    example_batch_inference()
    example_cascade_detection()
    example_attention_visualization()
    example_state_management()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
