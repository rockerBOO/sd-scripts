import pytest
import torch
from library.custom_train_functions import srpo_loss


@pytest.fixture
def mock_vae():
    """Create a mock VAE that tracks decode calls"""

    class MockVAE:
        def __init__(self):
            self.decode_call_count = 0
            self.batch_sizes_seen = []
            self.scaling_factor = 0.3611
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        def decode(self, latents):
            """Mock VAE decode that tracks batch sizes"""
            self.decode_call_count += 1
            self.batch_sizes_seen.append(latents.shape[0])
            # Return fake images [B, 3, H, W]
            batch_size = latents.shape[0]
            return torch.randn(batch_size, 3, 512, 512)

        def parameters(self):
            """Mock parameters for dtype detection"""
            param = torch.nn.Parameter(torch.tensor([1.0], dtype=self.dtype))
            yield param  # Use yield to return a generator/iterator

        def to(self, device):
            self.device = device
            return self

    return MockVAE()


@pytest.fixture
def mock_clip_reward_model():
    """Create a mock CLIP reward model"""

    class MockCLIPRewardModel:
        def __call__(self, images, prompts):
            """Return mock rewards"""
            batch_size = images.shape[0]
            return torch.randn(batch_size)

    return MockCLIPRewardModel()


@pytest.fixture
def reward_inputs_factory(mock_vae, mock_clip_reward_model):
    """Factory to create reward_inputs with different batch sizes"""

    def create_reward_inputs(batch_size=8):
        return {
            "latents_recovered": torch.randn(batch_size, 4, 64, 64),
            "sigma_t": torch.rand(batch_size, 1, 1, 1),
            "captions": [f"caption {i}" for i in range(batch_size)],
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "model_pred": torch.randn(batch_size, 4, 64, 64),
            "target": torch.randn(batch_size, 4, 64, 64),
        }

    return create_reward_inputs


class TestVAEBatchSizeConfig:
    def test_vae_default_batch_size(self, reward_inputs_factory, mock_vae):
        """Verify default VAE batch size is 4"""
        # Arrange: Create reward inputs with batch_size=8
        reward_inputs = reward_inputs_factory(batch_size=8)

        # Act: Call srpo_loss without vae_batch_size parameter (should default to 4)
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0)

        # Assert: VAE decode should be called 2 times (8/4=2)
        assert mock_vae.decode_call_count == 2
        assert mock_vae.batch_sizes_seen == [4, 4]

    def test_vae_custom_batch_size_2(self, reward_inputs_factory, mock_vae):
        """Test custom VAE batch_size=2"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=8)

        # Act: Use vae_batch_size=2
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=2)

        # Assert: VAE decode should be called 4 times (8/2=4)
        assert mock_vae.decode_call_count == 4
        assert mock_vae.batch_sizes_seen == [2, 2, 2, 2]

    def test_vae_custom_batch_size_8(self, reward_inputs_factory, mock_vae):
        """Test custom VAE batch_size=8"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=8)

        # Act: Use srpo_reward_vae_batch_size=8
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=8)

        # Assert: VAE decode should be called 1 time (8/8=1)
        assert mock_vae.decode_call_count == 1
        assert mock_vae.batch_sizes_seen == [8]

    def test_vae_custom_batch_size_1(self, reward_inputs_factory, mock_vae):
        """Test custom VAE batch_size=1"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=4)

        # Act: Use srpo_reward_vae_batch_size=1
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=1)

        # Assert: VAE decode should be called 4 times
        assert mock_vae.decode_call_count == 4
        assert mock_vae.batch_sizes_seen == [1, 1, 1, 1]


class TestBatchSplitting:
    def test_batch_size_larger_than_input(self, reward_inputs_factory, mock_vae):
        """When vae_batch_size >= input batch, decode all at once"""
        # Arrange: batch_size=4, srpo_reward_vae_batch_size=8
        reward_inputs = reward_inputs_factory(batch_size=4)

        # Act
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=8)

        # Assert: VAE decode should be called once with full batch
        assert mock_vae.decode_call_count == 1
        assert mock_vae.batch_sizes_seen == [4]

    def test_uneven_batch_splitting(self, reward_inputs_factory, mock_vae):
        """Test uneven batch splitting (7 images, batch_size=3)"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=7)

        # Act
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=3)

        # Assert: Should call with [3, 3, 1]
        assert mock_vae.decode_call_count == 3
        assert mock_vae.batch_sizes_seen == [3, 3, 1]

    def test_output_concatenation_correctness(self, reward_inputs_factory, mock_vae):
        """Verify output shapes are correct after batch splitting"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=10)

        # Act
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=3)

        # Assert: Loss should have correct batch size
        assert loss.shape[0] == 10
        # VAE should have been called 4 times: [3, 3, 3, 1]
        assert mock_vae.decode_call_count == 4
        assert sum(mock_vae.batch_sizes_seen) == 10


class TestEdgeCases:
    def test_batch_size_one_input(self, reward_inputs_factory, mock_vae):
        """Test with single image input"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=1)

        # Act
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=4)

        # Assert: Should decode once with batch_size=1
        assert mock_vae.decode_call_count == 1
        assert mock_vae.batch_sizes_seen == [1]

    def test_vae_batch_size_validation(self, reward_inputs_factory):
        """Test that invalid vae_batch_size raises appropriate error"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=4)

        # Act & Assert: vae_batch_size must be positive
        with pytest.raises(ValueError, match="vae_batch_size must be positive"):
            srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=0)

        with pytest.raises(ValueError, match="vae_batch_size must be positive"):
            srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=-1)


class TestMemoryImpact:
    def test_vae_batch_size_affects_intermediate_memory(self, reward_inputs_factory, mock_vae):
        """Verify smaller batch sizes use less intermediate memory"""
        # This test verifies the logic, not actual memory usage
        # Smaller batches = more decode calls but smaller intermediate tensors

        # Arrange
        reward_inputs_large = reward_inputs_factory(batch_size=16)
        reward_inputs_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in reward_inputs_large.items()}

        # Reset mock
        mock_vae.decode_call_count = 0
        mock_vae.batch_sizes_seen = []

        # Act: Large batch size (decode all at once)
        loss_large, _ = srpo_loss(reward_inputs=reward_inputs_large, beta=1.0, vae_batch_size=16)

        # Reset mock
        mock_vae.decode_call_count = 0
        mock_vae.batch_sizes_seen = []

        # Act: Small batch size (decode in chunks)
        loss_small, _ = srpo_loss(reward_inputs=reward_inputs_copy, beta=1.0, vae_batch_size=2)

        # Assert: More calls with smaller batches
        assert mock_vae.decode_call_count == 8  # 16/2 = 8 calls
        assert all(size == 2 for size in mock_vae.batch_sizes_seen)

    def test_metrics_include_vae_batch_size(self, reward_inputs_factory):
        """Verify metrics include vae_batch_size used"""
        # Arrange
        reward_inputs = reward_inputs_factory(batch_size=8)

        # Act
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0, vae_batch_size=2)

        # Assert: Metrics should include vae_batch_size
        assert "vram/vae_batch_size" in metrics
        assert metrics["vram/vae_batch_size"] == 2
