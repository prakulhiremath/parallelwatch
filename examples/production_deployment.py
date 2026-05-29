"""
Production Deployment Example: Real-Time Infrastructure Monitoring
Demonstrates ParallelWatch deployment for high-frequency metric streams.
"""

import torch
import numpy as np
import time
from parallelwatch import ParallelWatchEngine, EngineConfig
from parallelwatch.utils import StreamNormalizer


def benchmark_streaming_latency():
    """Benchmark single-step streaming latency."""
    print("=" * 70)
    print("Latency Benchmark: Streaming Mode")
    print("=" * 70)
    
    configs = [
        (10, 64),
        (50, 128),
        (100, 256),
        (500, 512),
        (1000, 1024),
    ]
    
    results = []
    
    for num_metrics, hidden_dim in configs:
        config = EngineConfig(
            num_metrics=num_metrics,
            hidden_dim=hidden_dim
        )
        engine = ParallelWatchEngine(config)
        engine.reset_state()
        
        warmup_steps = 100
        benchmark_steps = 1000
        
        for _ in range(warmup_steps):
            engine.step(torch.randn(num_metrics))
        
        start = time.perf_counter()
        for _ in range(benchmark_steps):
            engine.step(torch.randn(num_metrics))
        elapsed = time.perf_counter() - start
        
        latency_us = (elapsed / benchmark_steps) * 1e6
        throughput = benchmark_steps / elapsed
        
        results.append({
            "metrics": num_metrics,
            "hidden_dim": hidden_dim,
            "latency_us": latency_us,
            "throughput": throughput,
        })
        
        print(f"Metrics: {num_metrics:4d} | Hidden: {hidden_dim:4d} | "
              f"Latency: {latency_us:7.2f}μs | Throughput: {throughput:10.0f} steps/s")
    
    return results


def benchmark_batch_inference():
    """Benchmark batch processing throughput."""
    print("\n" + "=" * 70)
    print("Throughput Benchmark: Batch Inference")
    print("=" * 70)
    
    batch_sizes = [1, 8, 16, 32, 64]
    sequence_length = 64
    num_metrics = 100
    
    config = EngineConfig(
        num_metrics=num_metrics,
        hidden_dim=256
    )
    engine = ParallelWatchEngine(config)
    
    results = []
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, sequence_length, num_metrics, 1)
        
        warmup_runs = 10
        benchmark_runs = 100
        
        for _ in range(warmup_runs):
            engine.forward(x)
        
        start = time.perf_counter()
        for _ in range(benchmark_runs):
            engine.forward(x)
        elapsed = time.perf_counter() - start
        
        samples_per_second = (batch_size * sequence_length * benchmark_runs) / elapsed
        
        results.append({
            "batch_size": batch_size,
            "throughput": samples_per_second,
        })
        
        print(f"Batch: {batch_size:2d} | Samples/sec: {samples_per_second:12.0f}")
    
    return results


def benchmark_memory_usage():
    """Estimate memory consumption."""
    print("\n" + "=" * 70)
    print("Memory Usage Analysis")
    print("=" * 70)
    
    configs = [
        (10, 64),
        (100, 256),
        (1000, 512),
        (10000, 1024),
    ]
    
    for num_metrics, hidden_dim in configs:
        config = EngineConfig(
            num_metrics=num_metrics,
            hidden_dim=hidden_dim
        )
        engine = ParallelWatchEngine(config)
        
        total_params = sum(p.numel() for p in engine.parameters())
        buffers_size = sum(b.numel() for b in engine.buffers())
        
        state_size = num_metrics * hidden_dim
        total_tensors = total_params + buffers_size + state_size
        
        memory_mb = (total_tensors * 4) / (1024 ** 2)
        
        print(f"Metrics: {num_metrics:5d} | Hidden: {hidden_dim:4d} | "
              f"Memory: {memory_mb:8.2f}MB | Params: {total_params:10d}")


def production_monitoring_loop():
    """Simulate production monitoring loop."""
    print("\n" + "=" * 70)
    print("Production Monitoring Loop Simulation")
    print("=" * 70)
    
    config = EngineConfig(
        num_metrics=200,
        hidden_dim=512,
        num_attention_heads=8
    )
    engine = ParallelWatchEngine(config)
    engine.reset_state()
    
    normalizer = StreamNormalizer(num_metrics=200, momentum=0.995)
    
    alert_threshold = 0.75
    cascade_threshold = 0.70
    
    anomaly_count = 0
    cascade_count = 0
    
    print(f"Processing 10000 timesteps with {config.num_metrics} metrics...")
    print(f"Alert threshold: {alert_threshold:.2f}")
    print(f"Cascade threshold: {cascade_threshold:.2f}\n")
    
    start_time = time.perf_counter()
    
    for step in range(10000):
        if step < 3000:
            x_t = np.random.normal(0, 0.1, 200)
        elif step < 6000:
            x_t = np.random.normal(0, 0.1, 200)
            x_t[10:20] += np.random.normal(1.0, 0.3, 10)
        else:
            x_t = np.random.normal(0, 0.1, 200)
        
        normalizer.update(x_t)
        x_normalized = normalizer.normalize(x_t)
        
        output = engine.step(torch.FloatTensor(x_normalized))
        
        anomaly_scores = output["anomaly_scores"]
        cascade_score = output["cascade_score"].item()
        
        high_anomaly = (anomaly_scores > alert_threshold).sum().item()
        
        if cascade_score > cascade_threshold:
            cascade_count += 1
            if step % 500 == 0:
                print(f"Step {step}: CASCADE ALERT (score={cascade_score:.3f})")
        
        if high_anomaly > 10:
            anomaly_count += 1
            if step % 500 == 0:
                print(f"Step {step}: ANOMALY ALERT ({high_anomaly} metrics)")
    
    elapsed = time.perf_counter() - start_time
    
    print(f"\nResults:")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Anomaly alerts: {anomaly_count}")
    print(f"Cascade alerts: {cascade_count}")
    print(f"Throughput: {10000 / elapsed:.0f} steps/s")


def distributed_metric_groups():
    """Simulate monitoring across multiple metric groups (e.g., per-service)."""
    print("\n" + "=" * 70)
    print("Multi-Group Monitoring")
    print("=" * 70)
    
    services = ["frontend", "api", "database", "cache", "queue"]
    metrics_per_service = 50
    
    engines = {}
    
    for service in services:
        config = EngineConfig(
            num_metrics=metrics_per_service,
            hidden_dim=128
        )
        engines[service] = ParallelWatchEngine(config)
        engines[service].reset_state()
    
    print(f"Initialized {len(engines)} service monitors ({metrics_per_service} metrics each)")
    
    total_metrics = len(services) * metrics_per_service
    print(f"Total metrics: {total_metrics}")
    print(f"Total independent SSM states: {total_metrics}\n")
    
    for step in range(100):
        for service in services:
            x_t = torch.randn(metrics_per_service) * 0.1
            
            if step > 50 and service == "database":
                x_t[10:15] += 2.0
            
            output = engines[service].step(x_t)
            
            if output["cascade_score"] > 0.7:
                print(f"Step {step}: CASCADE in {service}")


def checkpoint_restore():
    """Demonstrate checkpoint save/restore for fault tolerance."""
    print("\n" + "=" * 70)
    print("Checkpoint Save/Restore")
    print("=" * 70)
    
    config = EngineConfig(
        num_metrics=100,
        hidden_dim=256
    )
    engine = ParallelWatchEngine(config)
    engine.reset_state()
    
    print("Processing 100 steps...")
    for step in range(100):
        x_t = torch.randn(100) * 0.1
        engine.step(x_t)
    
    checkpoint_state = engine.get_state()
    checkpoint_counter = engine.step_counter.clone()
    
    print(f"Checkpoint saved at step {checkpoint_counter.item()}")
    print(f"State shape: {checkpoint_state.shape}")
    
    print("Processing 50 more steps...")
    for step in range(50):
        x_t = torch.randn(100) * 0.1
        engine.step(x_t)
    
    print(f"Current step: {engine.step_counter.item()}")
    
    engine.set_state(checkpoint_state)
    print(f"Restored to step {checkpoint_counter.item()}")
    print(f"State matches checkpoint: {torch.allclose(engine.get_state(), checkpoint_state)}")


if __name__ == "__main__":
    print("\nParallelWatch Production Benchmarks & Deployment Examples")
    print("=" * 70)
    
    benchmark_streaming_latency()
    benchmark_batch_inference()
    benchmark_memory_usage()
    production_monitoring_loop()
    distributed_metric_groups()
    checkpoint_restore()
    
    print("\n" + "=" * 70)
    print("All benchmarks completed!")
    print("=" * 70)
