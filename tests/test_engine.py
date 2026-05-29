"""
Unit tests for ParallelWatch Engine.
"""

import torch
import pytest
import numpy as np
from parallelwatch import ParallelWatchEngine, EngineConfig
from parallelwatch.utils import (
    StreamNormalizer, create_synthetic_cascade, batch_data,
    compute_anomaly_metrics, moving_average_filter
)


class TestEngineConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        config = EngineConfig(num_metrics=10, hidden_dim=64)
        assert config.num_metrics == 10
        assert config.hidden_dim == 64
        assert config.num_attention_heads == 4
        assert config.dropout == 0.0
    
    def test_config_validation(self):
        config = EngineConfig(
            num_metrics=100,
            hidden_dim=128,
            num_attention_heads=8,
            dropout=0.1,
            decay_rate_min=0.85,
            decay_rate_max=0.99
        )
        assert config.num_metrics == 100


class TestParallelWatchEngine:
    """Test core engine functionality."""
    
    @pytest.fixture
    def engine_config(self):
        return EngineConfig(
            num_metrics=10,
            hidden_dim=64,
            num_attention_heads=4,
            dropout=0.0
        )
    
    @pytest.fixture
    def engine(self, engine_config):
        return ParallelWatchEngine(engine_config)
    
    def test_engine_initialization(self, engine):
        assert engine.num_metrics == 10
        assert engine.hidden_dim == 64
        assert engine.hidden_state.shape == (1, 10, 64)
    
    def test_ssm_step_single_batch(self, engine):
        x_t = torch.randn(1, 10, 1)
        h_prev = engine.hidden_state
        
        h_new, x_recon = engine._ssm_step(x_t, h_prev)
        
        assert h_new.shape == (1, 10, 64)
        assert x_recon.shape == (1, 10, 1)
    
    def test_ssm_step_multi_batch(self, engine):
        batch_size = 8
        x_t = torch.randn(batch_size, 10, 1)
        h_prev = torch.randn(batch_size, 10, 64)
        
        h_new, x_recon = engine._ssm_step(x_t, h_prev)
        
        assert h_new.shape == (batch_size, 10, 64)
        assert x_recon.shape == (batch_size, 10, 1)
    
    def test_decay_matrix_stability(self, engine):
        decay = engine._get_decay_matrix()
        
        assert decay.shape == (10, 64)
        assert torch.all(decay > 0)
        assert torch.all(decay < 1)
    
    def test_anomaly_score_computation(self, engine):
        x_t = torch.randn(4, 10, 1)
        x_recon = torch.randn(4, 10, 1)
        h_t = torch.randn(4, 10, 64)
        
        scores = engine._compute_anomaly_scores(x_t, x_recon, h_t)
        
        assert scores.shape == (4, 10)
        assert torch.all(scores >= 0)
    
    def test_cascade_score_computation(self, engine):
        h_t = torch.randn(4, 10, 64)
        attention = torch.softmax(torch.randn(4, 10, 10), dim=-1)
        
        cascade = engine._compute_cascade_score(h_t, attention)
        
        assert cascade.shape == (4,)
    
    def test_forward_full_sequence(self, engine):
        batch_size, time_steps, num_metrics = 4, 32, 10
        x = torch.randn(batch_size, time_steps, num_metrics, 1)
        
        output = engine.forward(x, return_states=True)
        
        assert output["anomaly_scores"].shape == (batch_size, time_steps, num_metrics)
        assert output["cascade_scores"].shape == (batch_size, time_steps)
        assert output["attention_weights"].shape == (batch_size, time_steps, num_metrics, num_metrics)
        assert output["hidden_states"].shape == (batch_size, time_steps, num_metrics, 64)
    
    def test_forward_without_states(self, engine):
        x = torch.randn(2, 16, 10, 1)
        output = engine.forward(x, return_states=False)
        
        assert "hidden_states" not in output
        assert output["anomaly_scores"].shape == (2, 16, 10)
    
    def test_streaming_step_single_metric(self, engine):
        engine.reset_state()
        x_t = torch.randn(10)
        
        output = engine.step(x_t)
        
        assert output["anomaly_scores"].shape == (10,)
        assert isinstance(output["cascade_score"], torch.Tensor)
        assert output["attention_weights"].shape == (10, 10)
    
    def test_streaming_step_batch(self, engine):
        engine.reset_state()
        x_t = torch.randn(4, 10)
        
        output = engine.step(x_t)
        
        assert output["anomaly_scores"].shape == (4, 10)
        assert output["cascade_score"].shape == (4,)
        assert output["attention_weights"].shape == (4, 10, 10)
    
    def test_state_persistence(self, engine):
        engine.reset_state()
        x1 = torch.randn(1, 10, 1)
        x2 = torch.randn(1, 10, 1)
        
        state1 = engine.hidden_state.clone()
        engine.step(x1.squeeze(-1))
        state2 = engine.hidden_state.clone()
        
        assert not torch.allclose(state1, state2)
    
    def test_state_reset(self, engine):
        engine.step(torch.randn(10))
        state_before = engine.hidden_state.clone()
        
        engine.reset_state()
        state_after = engine.hidden_state.clone()
        
        assert not torch.allclose(state_before, state_after)
        assert torch.allclose(state_after, torch.zeros_like(state_after))
    
    def test_state_getset(self, engine):
        new_state = torch.randn(1, 10, 64)
        engine.set_state(new_state)
        
        retrieved = engine.get_state()
        assert torch.allclose(retrieved, new_state)
    
    def test_gradient_flow(self, engine):
        engine.train()
        x = torch.randn(2, 8, 10, 1, requires_grad=True)
        output = engine.forward(x)
        
        loss = output["anomaly_scores"].mean() + output["cascade_scores"].mean()
        loss.backward()
        
        assert engine.decay_log.grad is not None
        assert engine.input_proj.grad is not None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_stream_normalizer(self):
        normalizer = StreamNormalizer(num_metrics=5)
        data = np.random.normal(10, 2, (100, 5))
        
        for sample in data:
            normalizer.update(sample)
        
        normalized = normalizer.normalize(data[:10])
        assert normalized.shape == (10, 5)
        assert np.abs(normalized.mean()) < 0.1
    
    def test_synthetic_cascade(self):
        data, labels = create_synthetic_cascade(
            num_metrics=10,
            sequence_length=100,
            cascade_start=30,
            cascade_end=60,
            cascade_indices=[0, 1, 2]
        )
        
        assert data.shape == (100, 10)
        assert labels.shape == (100, 10)
        assert labels[50, 0] == 1
        assert labels[15, 0] == 0
    
    def test_batch_data(self):
        data = torch.randn(200, 10)
        batches = batch_data(data, batch_size=4, sequence_length=16)
        
        assert len(batches) > 0
        for batch in batches:
            assert batch.size(-1) == 1
    
    def test_moving_average_filter(self):
        x = torch.randn(100)
        filtered = moving_average_filter(x, window_size=5)
        
        assert filtered.shape == x.shape
    
    def test_anomaly_metrics(self):
        scores = torch.sigmoid(torch.randn(100))
        labels = torch.randint(0, 2, (100,))
        
        metrics = compute_anomaly_metrics(scores, labels, threshold=0.5)
        
        assert "auroc" in metrics or "warning" in metrics
        assert "ap" in metrics or "warning" in metrics


class TestProductionScenarios:
    """Test production-like scenarios."""
    
    def test_production_streaming_loop(self):
        config = EngineConfig(num_metrics=50, hidden_dim=128)
        engine = ParallelWatchEngine(config)
        engine.reset_state()
        
        results = []
        for _ in range(1000):
            x_t = torch.randn(50)
            output = engine.step(x_t)
            results.append(output)
        
        assert len(results) == 1000
        assert all("anomaly_scores" in r for r in results)
    
    def test_production_batch_processing(self):
        config = EngineConfig(num_metrics=100, hidden_dim=256)
        engine = ParallelWatchEngine(config)
        
        batch_size = 16
        time_steps = 64
        x = torch.randn(batch_size, time_steps, 100, 1)
        
        output = engine.forward(x)
        
        assert output["anomaly_scores"].shape == (batch_size, time_steps, 100)
        assert torch.all(torch.isfinite(output["cascade_scores"]))
    
    def test_gpu_compatibility(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = EngineConfig(num_metrics=50, hidden_dim=128, device="cuda")
        engine = ParallelWatchEngine(config).cuda()
        
        x = torch.randn(4, 32, 50, 1).cuda()
        output = engine.forward(x)
        
        assert output["anomaly_scores"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
