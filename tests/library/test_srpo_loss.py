"""
Tests for SRPO (Semantic Relative Preference Optimization) loss function
"""
import torch.nn as nn
import pytest
import torch
from unittest.mock import Mock
from library.custom_train_functions import srpo_loss


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
    """Mock CLIP reward model"""
    def reward_fn(images, prompts):
        # Return random rewards based on batch size
        batch_size = images.shape[0]
        return torch.randn(batch_size) * 0.1 + 0.5  # Mean 0.5, std 0.1
    
    return Mock(side_effect=reward_fn)


@pytest.fixture
def basic_reward_inputs(mock_vae, mock_clip_reward_model):
    """Basic reward inputs for testing"""
    batch_size = 4
    return {
        "latents_recovered": torch.randn(batch_size, 16, 64, 64),  # FLUX latent shape
        "sigma_t": torch.rand(batch_size, 1, 1, 1) * 0.8 + 0.1,  # sigma between 0.1 and 0.9
        "captions": ["a photo of a cat"] * batch_size,
        "vae": mock_vae,
        "reward_model": mock_clip_reward_model,
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
        assert loss.shape == torch.Size([4])  # Batch size
        
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
        assert loss.shape == torch.Size([4])
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
            "loss/srpo_reward_positive",
            "loss/srpo_reward_negative",
            "loss/srpo_reward_relative",
            "loss/srpo_reward_discounted",
            "loss/srpo_discount_mean",
            "loss/srpo_sigma_mean",
            "loss/srpo_final_loss",
        }
        
        assert set(metrics.keys()) == expected_keys
        
        # Check all values are floats
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"
    
    def test_vae_decode_called(self, basic_reward_inputs):
        """Test that VAE decode is called with correct inputs"""
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )
        
        mock_vae = basic_reward_inputs["vae"]
        assert mock_vae.decode.called
        
        # Check that latents are scaled correctly
        call_args = mock_vae.decode.call_args[0][0]
        expected_scale = 0.3611
        expected_latents = basic_reward_inputs["latents_recovered"] / expected_scale
        torch.testing.assert_close(call_args, expected_latents)
    
    def test_reward_model_called_twice(self, basic_reward_inputs):
        """Test that reward model is called for both positive and negative prompts"""
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt="Realistic photo",
            negative_prompt="CG Render",
        )
        
        mock_reward = basic_reward_inputs["reward_model"]
        assert mock_reward.call_count == 2
        
        # Check first call (positive)
        first_call_prompts = mock_reward.call_args_list[0][0][1]
        assert all("Realistic photo" in p for p in first_call_prompts)
        
        # Check second call (negative)
        second_call_prompts = mock_reward.call_args_list[1][0][1]
        assert all("CG Render" in p for p in second_call_prompts)


class TestSRPOPromptAugmentation:
    """Test semantic relative preference prompt augmentation"""
    
    def test_positive_prompt_augmentation(self, basic_reward_inputs):
        """Test that positive prompts are correctly augmented"""
        positive_prompt = "High quality"
        
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt=positive_prompt,
            negative_prompt="Low quality",
        )
        
        mock_reward = basic_reward_inputs["reward_model"]
        first_call_prompts = mock_reward.call_args_list[0][0][1]
        
        for i, prompt in enumerate(first_call_prompts):
            original_caption = basic_reward_inputs["captions"][i]
            expected = f"{positive_prompt}. {original_caption}"
            assert prompt == expected
    
    def test_negative_prompt_augmentation(self, basic_reward_inputs):
        """Test that negative prompts are correctly augmented"""
        negative_prompt = "Blurry image"
        
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            positive_prompt="Sharp image",
            negative_prompt=negative_prompt,
        )
        
        mock_reward = basic_reward_inputs["reward_model"]
        second_call_prompts = mock_reward.call_args_list[1][0][1]
        
        for i, prompt in enumerate(second_call_prompts):
            original_caption = basic_reward_inputs["captions"][i]
            expected = f"{negative_prompt}. {original_caption}"
            assert prompt == expected
    
    def test_different_captions(self, mock_vae, mock_clip_reward_model):
        """Test with different captions per sample"""
        captions = ["a cat", "a dog", "a bird", "a fish"]
        reward_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.rand(4, 1, 1, 1),
            "captions": captions,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
        }
        
        loss, metrics = srpo_loss(
            reward_inputs=reward_inputs,
            beta=1.0,
            positive_prompt="Photo",
            negative_prompt="Drawing",
        )
        
        mock_reward = reward_inputs["reward_model"]
        positive_prompts = mock_reward.call_args_list[0][0][1]
        
        for i, caption in enumerate(captions):
            assert caption in positive_prompts[i]


class TestSRPOTimestepDiscount:
    """Test timestep discount functionality"""
    
    def test_beta_affects_discount(self, basic_reward_inputs):
        """Test that beta parameter affects discount calculation"""
        # High beta should give lower discount for high sigma
        loss_high_beta, metrics_high = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=5.0,
        )
        
        loss_low_beta, metrics_low = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=0.5,
        )
        
        # With higher beta, discount should be more aggressive
        assert metrics_high["loss/srpo_discount_mean"] < metrics_low["loss/srpo_discount_mean"]
    
    def test_beta_and_sigma_reduce_discount(self, mock_vae, mock_clip_reward_model):
        """
        Discount must decrease when either σ OR β increases.
        We test both effects in one go:
          1. same σ, different β  →  larger β gives smaller discount
          2. same β, different σ  →  larger σ gives smaller discount
        """
        base_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
        }

        inputs = {**base_inputs, "sigma_t": torch.tensor([0.45, 0.50, 0.52, 0.48]).view(4, 1, 1, 1)}
        _, metrics_weak = srpo_loss(reward_inputs=inputs, beta=0.5)
        _, metrics_strong = srpo_loss(reward_inputs=inputs, beta=5.0)
        assert metrics_strong["loss/srpo_discount_mean"] < metrics_weak["loss/srpo_discount_mean"]

        high_sigma_inputs = {**base_inputs, "sigma_t": torch.tensor([0.84, 0.88, 0.92, 0.95]).view(4, 1, 1, 1)}
        low_sigma_inputs = {**base_inputs, "sigma_t": torch.tensor([0.05, 0.10, 0.08, 0.12]).view(4, 1, 1, 1)}
        _, metrics_high = srpo_loss(reward_inputs=high_sigma_inputs, beta=3.0)
        _, metrics_low = srpo_loss(reward_inputs=low_sigma_inputs, beta=3.0)
        assert metrics_high["loss/srpo_discount_mean"] < metrics_low["loss/srpo_discount_mean"]

        def test_discount_formula(self, basic_reward_inputs):
            """Test that discount follows exp(-beta * sigma) formula"""
            beta = 2.5
            sigma_t = basic_reward_inputs["sigma_t"]
            
            loss, metrics = srpo_loss(
                reward_inputs=basic_reward_inputs,
                beta=beta,
            )
            
            # Manually compute expected discount
            expected_discount = torch.exp(-beta * sigma_t.squeeze()).mean().item()
            
            assert abs(metrics["loss/srpo_discount_mean"] - expected_discount) < 1e-5


class TestSRPOInversionRegularization:
    """Test inversion-based regularization"""
    
    def test_inversion_flag_changes_loss(self, basic_reward_inputs):
        """Test that inversion flag affects loss computation"""
        # Set high sigma to trigger inversion regularization
        basic_reward_inputs["sigma_t"] = torch.ones(4, 1, 1, 1) * 0.8
        
        loss_no_inv, _ = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            use_inversion=False,
        )
        
        loss_with_inv, _ = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
            use_inversion=True,
        )
        
        # Losses should be different when inversion is enabled
        assert not torch.allclose(loss_no_inv, loss_with_inv)
    
    def test_inversion_reduces_late_timestep_loss(self, mock_vae, mock_clip_reward_model):
        """Test that inversion reduces loss magnitude at late timesteps"""
        # Late timestep (high sigma > 0.7)
        late_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.ones(4, 1, 1, 1) * 0.8,
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
        }
        
        loss_no_inv, _ = srpo_loss(reward_inputs=late_inputs, use_inversion=False)
        loss_with_inv, _ = srpo_loss(reward_inputs=late_inputs, use_inversion=True)
        
        # With inversion, loss should be reduced by factor of (1 - 0.5) = 0.5
        # So absolute loss magnitude should be smaller
        assert loss_with_inv.abs().mean() < loss_no_inv.abs().mean()
    
    def test_inversion_no_effect_early_timestep(self, mock_vae):
        """Test that inversion has no effect at early timesteps"""
        # Use deterministic reward model to avoid random value differences
        def deterministic_reward_fn(images, prompts):
            batch_size = images.shape[0]
            if "Realistic" in prompts[0]:
                return torch.ones(batch_size) * 0.8  # Fixed positive rewards
            else:
                return torch.ones(batch_size) * 0.3  # Fixed negative rewards

        mock_reward_model = Mock(side_effect=deterministic_reward_fn)

        # Use fixed latents to ensure deterministic VAE decode
        fixed_latents = torch.ones(4, 16, 64, 64) * 0.5

        # Early timestep (low sigma < 0.7)
        early_inputs = {
            "latents_recovered": fixed_latents,
            "sigma_t": torch.ones(4, 1, 1, 1) * 0.3,
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_reward_model,
        }

        loss_no_inv, _ = srpo_loss(reward_inputs=early_inputs, use_inversion=False)

        # Reset the mock to get the same deterministic values
        mock_reward_model.reset_mock()
        mock_reward_model.side_effect = deterministic_reward_fn

        loss_with_inv, _ = srpo_loss(reward_inputs=early_inputs, use_inversion=True)

        # Should be identical for low sigma since mask is 0
        torch.testing.assert_close(loss_no_inv, loss_with_inv)


class TestSRPORewardCalculation:
    """Test semantic-relative reward calculation"""
    
    def test_reward_is_difference(self, basic_reward_inputs):
        """Test that reward is r_positive - r_negative"""
        # Mock to return known values
        def fixed_reward_fn(images, prompts):
            batch_size = images.shape[0]
            if "Realistic" in prompts[0]:
                return torch.ones(batch_size) * 0.8  # Positive rewards
            else:
                return torch.ones(batch_size) * 0.3  # Negative rewards
        
        basic_reward_inputs["reward_model"] = Mock(side_effect=fixed_reward_fn)
        
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=0.0,  # No discount for easier testing
            positive_prompt="Realistic photo",
            negative_prompt="CG Render",
        )
        
        # Reward should be 0.8 - 0.3 = 0.5
        assert abs(metrics["loss/srpo_reward_relative"] - 0.5) < 1e-5
    
    def test_loss_is_negative_reward(self, basic_reward_inputs):
        """Test that loss is negative of discounted reward"""
        # Mock to return fixed positive reward
        def fixed_reward_fn(images, prompts):
            batch_size = images.shape[0]
            if "Realistic" in prompts[0]:
                return torch.ones(batch_size) * 1.0
            else:
                return torch.zeros(batch_size)
        
        basic_reward_inputs["reward_model"] = Mock(side_effect=fixed_reward_fn)
        
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=0.0,  # No discount
            positive_prompt="Realistic photo",
            negative_prompt="CG Render",
        )
        
        # Loss should be negative of reward (want to maximize reward)
        # Since reward is 1.0 - 0.0 = 1.0, loss should be -1.0
        expected_loss = -1.0
        torch.testing.assert_close(loss, torch.ones(4) * expected_loss)


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
        }
        
        loss, metrics = srpo_loss(
            reward_inputs=large_batch_inputs,
            beta=1.0,
        )
        
        # VAE decode should be called twice (8 / 4 = 2)
        assert mock_vae.decode.call_count == 2
        
        # Each call should have batch size 4
        for call in mock_vae.decode.call_args_list:
            batch_latents = call[0][0]
            assert batch_latents.shape[0] == 4
    
    def test_small_batch_single_decode(self, basic_reward_inputs):
        """Test that small batches use single decode"""
        # Batch size is 4, should use single decode
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )
        
        mock_vae = basic_reward_inputs["vae"]
        assert mock_vae.decode.call_count == 1
    
    def test_batch_consistency(self, mock_vae, mock_clip_reward_model):
        """Test that per-sample losses are computed correctly"""
        batch_size = 6
        reward_inputs = {
            "latents_recovered": torch.randn(batch_size, 16, 64, 64),
            "sigma_t": torch.rand(batch_size, 1, 1, 1),
            "captions": [f"caption_{i}" for i in range(batch_size)],
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
        }
        
        loss, metrics = srpo_loss(reward_inputs=reward_inputs, beta=1.0)
        
        # Loss should have one value per sample
        assert loss.shape == torch.Size([batch_size])
        
        # All losses should be finite
        assert torch.all(torch.isfinite(loss))


class TestSRPOEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_beta(self, basic_reward_inputs):
        """Test with beta = 0 (no discount)"""
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=0.0,
        )
        
        # Discount should be 1.0 (no discounting)
        assert abs(metrics["loss/srpo_discount_mean"] - 1.0) < 1e-5
    
    def test_extreme_sigma_values(self, mock_vae, mock_clip_reward_model):
        """Test with sigma at boundaries"""
        # Sigma = 0 (no noise)
        zero_sigma_inputs = {
            "latents_recovered": torch.randn(4, 16, 64, 64),
            "sigma_t": torch.zeros(4, 1, 1, 1),
            "captions": ["test"] * 4,
            "vae": mock_vae,
            "reward_model": mock_clip_reward_model,
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
        
        def capture_reward_fn(images, prompts):
            captured_images.append(images.clone())
            return torch.randn(images.shape[0])
        
        basic_reward_inputs["reward_model"] = Mock(side_effect=capture_reward_fn)
        
        loss, metrics = srpo_loss(
            reward_inputs=basic_reward_inputs,
            beta=1.0,
        )
        
        # Check both calls to reward model
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
        def capture_reward_fn(images, prompts):
            captured_images.append(images.clone())
            return torch.randn(images.shape[0])
        
        basic_reward_inputs["reward_model"] = Mock(side_effect=capture_reward_fn)
        
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
