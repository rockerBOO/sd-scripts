"""
Tests for SRPO (Semantic Relative Preference Optimization) loss function
"""

import torch.nn as nn
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from library.custom_train_functions import srpo_loss
from library.reward_model import CLIPRewardModel
from accelerate import Accelerator


@pytest.fixture
def mock_vae():
    """Mock VAE decoder that behaves like a real one to dtype queries."""
    vae = Mock()
    vae.scaling_factor = 0.3611

    dummy = nn.Parameter(torch.empty(1, dtype=torch.float32))
    vae.parameters = lambda: iter([dummy])

    def decode_fn(latents):
        # return a tensor-like object (not a Mock) so .clamp, .to etc work
        return torch.randn(latents.shape[0], 3, 512, 512)

    vae.decode = Mock(side_effect=decode_fn)
    return vae


@pytest.fixture
def mock_clip_reward_model():
    """Mock CLIPRewardModel to return random rewards and similarities"""

    # Create a mock that will pass isinstance check
    mock_model = MagicMock(spec=CLIPRewardModel)

    def srp_cfg_side_effect(prompts, neg_prompt, images: torch.Tensor, k: float):
        batch_size = images.shape[0]
        reward = torch.randn(batch_size) * 0.1 + 0.5
        # Mock CLIP similarities (typically in range 0.2-0.4)
        pos_similarity = torch.randn(batch_size) * 0.05 + 0.25
        neg_similarity = torch.randn(batch_size) * 0.05 + 0.23
        return reward, pos_similarity, neg_similarity

    mock_model.SRP_cfg = Mock(side_effect=srp_cfg_side_effect)

    return mock_model


@pytest.fixture
def basic_reward_inputs(mock_vae, mock_clip_reward_model):
    """Fixture for basic SRPO loss inputs"""
    batch_size = 4

    return {
        "latents_recovered": torch.randn(batch_size, 4, 8, 8),
        "sigma_t": torch.rand(batch_size, 1, 1, 1),
        "captions": ["a photo of a cat"] * batch_size,
        "vae": mock_vae,
        "reward_model": mock_clip_reward_model,
        "target": torch.randn(batch_size, 4, 8, 8),
        "model_pred": torch.randn(batch_size, 4, 8, 8),
    }


class TestSRPOLossBasic:
    """Test basic SRPO loss functionality"""

    def test_loss_returns_correct_types(self, basic_reward_inputs):
        """Test that loss function returns correct types"""
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt="Realistic photo",
            negative_prompt="CG Render",
        )

        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        # Loss shape should be (B, C, H, W) = (4, 4, 8, 8)
        assert loss.shape == torch.Size([4, 4, 8, 8])

    def test_loss_has_gradients(self, basic_reward_inputs):
        """Test that loss can be used for optimization"""
        # SRPO computes rewards in no_grad contexts (intentional design)
        # The loss is a detached tensor that serves as a training signal
        # We test that it's a valid tensor that can be used in optimization

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt="Realistic photo",
            negative_prompt="CG Render",
        )

        # Loss should be a valid tensor
        assert isinstance(loss, torch.Tensor)
        # Loss shape should be (B, C, H, W) = (4, 4, 8, 8)
        assert loss.shape == torch.Size([4, 4, 8, 8])
        assert torch.all(torch.isfinite(loss))

        # Loss can be used in optimization (even without grad_fn)
        # It acts as a reward signal, not a differentiable computation
        total_loss = loss.mean()
        assert isinstance(total_loss.item(), float)

    def test_metrics_keys(self, basic_reward_inputs):
        """Test that all expected metrics are present"""
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )

        expected_keys = {
            "loss/srpo_reward",
            "loss/srpo_reward_discounted",
            "loss/srpo_discount_mean",
            "loss/srpo_sigma_mean",
            "loss/srpo_final_loss",
            "loss/srpo_discount_index",
            "loss/srpo_sigma_value",
            "loss/srpo_timestep_value",
            "loss/srpo_is_inversion",
            "loss/srpo_clip_pos_similarity",
            "loss/srpo_clip_neg_similarity",
            "loss/srpo_clip_similarity_gap",
            "vram/vae_batch_size",
        }

        assert set(metrics.keys()) == expected_keys

        # Check all values are floats (except timestep_value and vae_batch_size which are int)
        for key, value in metrics.items():
            if key in ["loss/srpo_timestep_value", "vram/vae_batch_size"]:
                assert isinstance(value, int), f"{key} should be int, got {type(value)}"
            else:
                assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_vae_decode_called(self, basic_reward_inputs):
        """Test that VAE decode is called with correct inputs"""
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            vae_batch_size=4,  # Set to 4 to decode full batch at once
        )

        mock_vae = basic_reward_inputs["vae"]
        assert mock_vae.decode.called

        # Check that latents are scaled correctly
        call_args = mock_vae.decode.call_args[0][0]
        expected_scale = 0.3611
        expected_latents = basic_reward_inputs["latents_recovered"] / expected_scale
        torch.testing.assert_close(call_args, expected_latents)




class TestSRPOTimestepDiscount:
    """Test timestep discount functionality using linear schedules"""

    def test_linear_discount_schedule(self, basic_reward_inputs):
        """Test that discount uses linear schedule based on timestep index"""
        # With default schedule: denoise 0.1→0.25, inversion 0.3→0.01
        # timestep_index=0 should give discount_denoise_start
        basic_reward_inputs["timestep_index"] = 0
        basic_reward_inputs["is_inversion"] = False

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            discount_denoise_start=0.1,
            discount_denoise_end=0.25,
        )

        # At index 0, should be at start of schedule
        assert abs(metrics["loss/srpo_discount_mean"] - 0.1) < 0.01

    def test_discount_increases_with_timestep(self, mock_vae, mock_clip_reward_model):
        """Test that discount increases as timestep index increases (denoise branch)"""
        base_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.rand(4, 1, 1, 1),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(4, 16, 64, 64),
            "model_pred": torch.randn(4, 16, 64, 64),
            "is_inversion": False,
        }

        # Early timestep
        inputs_early = {**base_inputs, "timestep_index": 10}
        _, metrics_early = srpo_loss(
            reward_inputs=inputs_early,
            discount_denoise_start=0.1,
            discount_denoise_end=0.25,
        )

        # Late timestep
        inputs_late = {**base_inputs, "timestep_index": 90}
        _, metrics_late = srpo_loss(
            reward_inputs=inputs_late,
            discount_denoise_start=0.1,
            discount_denoise_end=0.25,
        )

        # For denoise branch, discount increases from 0.1 to 0.25
        assert metrics_late["loss/srpo_discount_mean"] > metrics_early["loss/srpo_discount_mean"]

    def test_inversion_discount_decreases(self, mock_vae, mock_clip_reward_model):
        """Test that inversion branch discount decreases with timestep"""
        base_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.rand(4, 1, 1, 1),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(4, 16, 64, 64),
            "model_pred": torch.randn(4, 16, 64, 64),
            "is_inversion": True,
        }

        # Early timestep
        inputs_early = {**base_inputs, "timestep_index": 10}
        _, metrics_early = srpo_loss(
            reward_inputs=inputs_early,
            discount_inversion_start=0.3,
            discount_inversion_end=0.01,
        )

        # Late timestep
        inputs_late = {**base_inputs, "timestep_index": 90}
        _, metrics_late = srpo_loss(
            reward_inputs=inputs_late,
            discount_inversion_start=0.3,
            discount_inversion_end=0.01,
        )

        # For inversion branch, discount decreases from 0.3 to 0.01
        assert metrics_late["loss/srpo_discount_mean"] < metrics_early["loss/srpo_discount_mean"]


class TestSRPOInversionRegularization:
    """Test inversion-based regularization (prompt swapping)"""

    def test_inversion_flag_changes_prompts(self, basic_reward_inputs):
        """Test that inversion flag affects prompt order"""
        # Track which prompts are passed to reward model
        captured_calls = []

        def capture_srp_cfg(prompts, neg_prompts, images, k):
            captured_calls.append({"pos": prompts, "neg": neg_prompts})
            batch_size = images.shape[0]
            reward = torch.ones(batch_size) * 0.5
            pos_sim = torch.ones(batch_size) * 0.25
            neg_sim = torch.ones(batch_size) * 0.23
            return reward, pos_sim, neg_sim

        mock_reward_model = MagicMock(spec=CLIPRewardModel)
        mock_reward_model.SRP_cfg = Mock(side_effect=capture_srp_cfg)
        basic_reward_inputs["reward_model"] = mock_reward_model

        # Test with is_inversion=True (normal prompt order)
        basic_reward_inputs["is_inversion"] = True
        loss_inv, _ = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt="Good",
            negative_prompt="Bad",
        )

        # Inversion branch should use normal order: (positive, negative)
        assert "Good" in captured_calls[0]["pos"][0]
        assert "Bad" in captured_calls[0]["neg"][0]

        # Reset and test denoise branch
        captured_calls.clear()
        mock_reward_model.SRP_cfg.reset_mock()
        mock_reward_model.SRP_cfg.side_effect = capture_srp_cfg

        basic_reward_inputs["is_inversion"] = False
        loss_denoise, _ = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt="Good",
            negative_prompt="Bad",
        )

        # Denoise branch should SWAP: (negative, positive)
        assert "Bad" in captured_calls[0]["pos"][0]
        assert "Good" in captured_calls[0]["neg"][0]



class TestSRPORewardCalculation:
    """Test semantic-relative reward calculation"""

    def test_reward_used_in_loss(self, basic_reward_inputs):
        """Test that reward from SRP_cfg is used in loss calculation"""

        # Mock to return fixed reward
        def fixed_srp_cfg(prompts, neg_prompt, images, k):
            batch_size = images.shape[0]
            reward = torch.ones(batch_size) * 2.5  # Fixed reward
            pos_sim = torch.ones(batch_size) * 0.25
            neg_sim = torch.ones(batch_size) * 0.20
            return reward, pos_sim, neg_sim

        mock_reward_model = MagicMock(spec=CLIPRewardModel)
        mock_reward_model.SRP_cfg = Mock(side_effect=fixed_srp_cfg)
        basic_reward_inputs["reward_model"] = mock_reward_model
        basic_reward_inputs["timestep_index"] = 0  # First timestep for predictable discount

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            reward_scale=1.0,  # No scaling (default)
            discount_denoise_start=0.1,
        )

        # Reward should be 2.5 (no scaling)
        expected_reward = 2.5
        assert abs(metrics["loss/srpo_reward"] - expected_reward) < 1e-5

    def test_loss_is_negative_reward(self, basic_reward_inputs):
        """Test that loss is negative reward (pure reward-based optimization)"""

        # Mock to return fixed reward
        def fixed_srp_cfg(prompts, neg_prompt, images, k):
            batch_size = images.shape[0]
            reward = torch.ones(batch_size) * 1.0  # Fixed reward
            pos_sim = torch.ones(batch_size) * 0.30
            neg_sim = torch.ones(batch_size) * 0.20
            return reward, pos_sim, neg_sim

        mock_reward_model = MagicMock(spec=CLIPRewardModel)
        mock_reward_model.SRP_cfg = Mock(side_effect=fixed_srp_cfg)
        basic_reward_inputs["reward_model"] = mock_reward_model
        basic_reward_inputs["timestep_index"] = 0

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            reward_scale=1.0,  # No scaling
            discount_denoise_start=1.0,  # No discount (discount = 1.0)
            discount_denoise_end=1.0,
        )

        # Reward = 1.0, discount = 1.0, loss = -1.0 (pure reward optimization, no noise penalty)
        # Final loss mean should be close to -1.0
        assert abs(loss.mean().item() + 1.0) < 0.01


class TestSRPOBatchProcessing:
    """Test batch processing functionality"""

    def test_large_batch_splits_decode(self, mock_vae, mock_clip_reward_model):
        """Test that large batches are split for VAE decode"""
        large_batch_inputs = {
            "latents_recovered": torch.randn(8, 16, 64, 64),  # Batch size > 4
            "sigma_t": torch.rand(8, 1, 1, 1),
            "captions": ["test"] * 8,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(8, 16, 64, 64),
            "model_pred": torch.randn(8, 16, 64, 64),
        }

        loss, metrics = srpo_loss(
            reward_inputs=large_batch_inputs,
            beta=1.0,
            vae_batch_size=4,  # Explicitly set batch size for testing
        )

        # VAE decode should be called twice (8 / 4 = 2)
        assert mock_vae.decode.call_count == 2

        # Each call should have batch size 4
        for call in mock_vae.decode.call_args_list:
            batch_latents = call[0][0]
            assert batch_latents.shape[0] == 4

    def test_small_batch_single_decode(self, basic_reward_inputs):
        """Test that small batches use single decode"""
        # Batch size is 4, with vae_batch_size=4 should use single decode
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            vae_batch_size=4,  # Explicitly set to test single decode
        )

        mock_vae = basic_reward_inputs["vae"]
        assert mock_vae.decode.call_count == 1

    def test_default_batch_size_decodes_individually(self, mock_vae, mock_clip_reward_model):
        """Test that default vae_batch_size=1 decodes each sample individually"""
        batch_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.rand(4, 1, 1, 1),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(4, 16, 64, 64),
            "model_pred": torch.randn(4, 16, 64, 64),
        }

        loss, metrics = srpo_loss(
            reward_inputs=batch_inputs,
            beta=1.0,
            # Use default vae_batch_size=1
        )

        # With default batch size 1, should decode 4 times (4 / 1 = 4)
        assert mock_vae.decode.call_count == 4

        # Each call should have batch size 1
        for call in mock_vae.decode.call_args_list:
            batch_latents = call[0][0]
            assert batch_latents.shape[0] == 1

    def test_batch_consistency(self, mock_vae, mock_clip_reward_model):
        """Test that per-sample losses are computed correctly"""
        batch_size = 6
        reward_inputs = {
            "latents_recovered": torch.randn(batch_size, 16, 64, 64),
            "sigma_t": torch.rand(batch_size, 1, 1, 1),
            "captions": [f"caption_{i}" for i in range(batch_size)],
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(batch_size, 16, 64, 64),
            "model_pred": torch.randn(batch_size, 16, 64, 64),
        }

        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0)

        # Loss shape should be (B, C, H, W) = (6, 16, 64, 64)
        assert loss.shape == torch.Size([batch_size, 16, 64, 64])

        # All losses should be finite
        assert torch.all(torch.isfinite(loss))


class TestSRPOEdgeCases:
    """Test edge cases and error handling"""

    def test_linear_discount_at_start(self, basic_reward_inputs):
        """Test that discount at start of schedule matches discount_denoise_start"""
        basic_reward_inputs["timestep_index"] = 0
        basic_reward_inputs["is_inversion"] = False

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            discount_denoise_start=0.1,
            discount_denoise_end=0.25,
        )

        # At timestep 0, discount should be at start value
        assert abs(metrics["loss/srpo_discount_mean"] - 0.1) < 0.01

    def test_extreme_sigma_values(self, mock_vae, mock_clip_reward_model):
        """Test with sigma at boundaries"""
        # Sigma = 0 (no noise)
        zero_sigma_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.zeros(4, 1, 1, 1),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(4, 16, 64, 64),
            "model_pred": torch.randn(4, 16, 64, 64),
        }

        loss_zero, metrics_zero = srpo_loss(reward_inputs=zero_sigma_inputs, beta=3.0)
        assert torch.all(torch.isfinite(loss_zero))

        # Sigma = 1 (pure noise)
        one_sigma_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.ones(4, 1, 1, 1),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(4, 16, 64, 64),
            "model_pred": torch.randn(4, 16, 64, 64),
        }

        loss_one, metrics_one = srpo_loss(reward_inputs=one_sigma_inputs, beta=3.0)
        assert torch.all(torch.isfinite(loss_one))

    def test_extreme_latent_values(self, basic_reward_inputs):
        """Test with extreme latent values"""
        # Very large latents
        basic_reward_inputs["latents_recovered"] = torch.ones(4, 16, 64, 64) * 100

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )

        # Should still produce finite loss
        assert torch.all(torch.isfinite(loss))

    def test_empty_captions(self, mock_vae, mock_clip_reward_model):
        """Test with empty caption strings"""
        empty_caption_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.rand(4, 1, 1, 1),
            "captions": [""] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
            "target": torch.randn(4, 16, 64, 64),
            "model_pred": torch.randn(4, 16, 64, 64),
        }

        loss, metrics = srpo_loss(
            reward_inputs=empty_caption_inputs,
            beta=1.0,
            positive_prompt="Good",
            negative_prompt="Bad",
        )

        # Should still work, prompts will be "Good. " and "Bad. "
        assert torch.all(torch.isfinite(loss))


class TestSRPOImageNormalization:
    """Test image normalization for CLIP"""

    def test_images_normalized_to_zero_one(self, basic_reward_inputs):
        """Test that decoded images are normalized to [0, 1]"""
        # Track what images are passed to reward model
        captured_images = []

        def capture_srp_cfg(prompts, neg_prompt, images, k):
            captured_images.append(images.clone())
            batch_size = images.shape[0]
            reward = torch.randn(batch_size)
            pos_sim = torch.randn(batch_size) * 0.05 + 0.25
            neg_sim = torch.randn(batch_size) * 0.05 + 0.23
            return reward, pos_sim, neg_sim

        mock_reward_model = MagicMock(spec=CLIPRewardModel)
        mock_reward_model.SRP_cfg = Mock(side_effect=capture_srp_cfg)
        basic_reward_inputs["reward_model"] = mock_reward_model

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )

        # SRP_cfg is called once, check images are normalized
        assert len(captured_images) == 1
        for images in captured_images:
            assert torch.all(images >= 0.0)
            assert torch.all(images <= 1.0)

    def test_images_clamped(self, basic_reward_inputs):
        """Test that extreme values are clamped"""

        # Mock VAE to return extreme values
        def extreme_decode(latents):
            # Return values outside [-1, 1] before normalization
            return torch.randn(latents.shape[0], 3, 512, 512) * 5

        basic_reward_inputs["vae"].decode = Mock(side_effect=extreme_decode)

        captured_images = []

        def capture_srp_cfg(prompts, neg_prompt, images, k):
            captured_images.append(images.clone())
            batch_size = images.shape[0]
            reward = torch.randn(batch_size)
            pos_sim = torch.randn(batch_size) * 0.05 + 0.25
            neg_sim = torch.randn(batch_size) * 0.05 + 0.23
            return reward, pos_sim, neg_sim

        mock_reward_model = MagicMock(spec=CLIPRewardModel)
        mock_reward_model.SRP_cfg = Mock(side_effect=capture_srp_cfg)
        basic_reward_inputs["reward_model"] = mock_reward_model

        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )

        # Even with extreme values, should be clamped to [0, 1]
        for images in captured_images:
            assert torch.all(images >= 0.0)
            assert torch.all(images <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
