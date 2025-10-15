"""
Integration tests for SRPO with training pipeline
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock 
from library.custom_train_functions import PreferenceOptimization, srpo_loss

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


class TestSRPOPreferenceOptimizationIntegration:
    """Test SRPO integration with PreferenceOptimization class"""
    
    @pytest.fixture
    def srpo_args(self):
        """Mock args for SRPO"""
        args = Mock()
        args.srpo_beta = 2.5
        args.srpo_positive_prompt = "High quality photo"
        args.srpo_negative_prompt = "Low quality render"
        args.srpo_use_inversion = False
        
        # Ensure other PO methods are None
        args.ddo_beta = None
        args.ddo_alpha = None
        args.bpo_beta = None
        args.bpo_lambda = None
        args.beta_dpo = None
        args.sdpo_beta = None
        args.mapo_beta = None
        args.simpo_beta = None
        args.cpo_beta = None
        
        return args
    
    def test_srpo_initialization(self, srpo_args):
        """Test that SRPO is properly initialized"""
        po = PreferenceOptimization(srpo_args)
        
        assert po.algo == "SRPO"
        assert po.reward_fn == srpo_loss
        assert po.args["beta"] == 2.5
        assert po.args["positive_prompt"] == "High quality photo"
        assert po.args["negative_prompt"] == "Low quality render"
        assert po.args["use_inversion"] is False
    
    def test_is_po_returns_true(self, srpo_args):
        """Test that is_po() returns True for SRPO"""
        po = PreferenceOptimization(srpo_args)
        assert po.is_po() is True
    
    def test_is_reward_based_returns_true(self, srpo_args):
        """Test that is_reward_based() returns True for SRPO"""
        po = PreferenceOptimization(srpo_args)
        assert po.is_reward_based() is True
    
    def test_is_reference_returns_false(self, srpo_args):
        """Test that is_reference() returns False for SRPO"""
        po = PreferenceOptimization(srpo_args)
        assert po.is_reference() is False
    
    def test_call_with_reward_inputs(self, srpo_args, mock_vae):
        """Test calling PreferenceOptimization with reward_inputs"""
        po = PreferenceOptimization(srpo_args)
        
        mock_reward_model = Mock(return_value=torch.randn(2))
        
        reward_inputs = {
            "latents_recovered": torch.randn(2, 16, 64, 64),
            "sigma_t": torch.rand(2, 1, 1, 1),
            "captions": ["test1", "test2"],
            "vae": mock_vae,
            "reward_model": mock_reward_model,
        }
        
        loss, metrics = po(reward_inputs=reward_inputs)
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert loss.shape == torch.Size([2])
    
    def test_call_requires_reward_inputs(self, srpo_args):
        """Test that calling without reward_inputs raises error"""
        po = PreferenceOptimization(srpo_args)
        
        with pytest.raises(AssertionError, match="SRPO requires reward_inputs"):
            po()
    
    def test_srpo_with_inversion(self):
        """Test SRPO with inversion enabled"""
        args = Mock()
        args.srpo_beta = 3.0
        args.srpo_positive_prompt = "Sharp"
        args.srpo_negative_prompt = "Blurry"
        args.srpo_use_inversion = True
        
        # Set other PO methods to None
        for attr in ['ddo_beta', 'ddo_alpha', 'bpo_beta', 'bpo_lambda', 
                     'beta_dpo', 'sdpo_beta', 'mapo_beta', 'simpo_beta', 'cpo_beta']:
            setattr(args, attr, None)
        
        po = PreferenceOptimization(args)
        
        assert po.args["use_inversion"] is True


class TestSRPONotInitializedWhenOtherMethodsUsed:
    """Test that SRPO doesn't interfere with other PO methods"""
    
    def test_dpo_not_srpo(self):
        """Test that DPO doesn't trigger SRPO"""
        args = Mock()
        args.beta_dpo = 0.1
        args.srpo_beta = None
        args.ddo_beta = None
        args.ddo_alpha = None
        args.bpo_beta = None
        args.bpo_lambda = None
        args.sdpo_beta = None
        args.mapo_beta = None
        args.simpo_beta = None
        args.cpo_beta = None
        
        po = PreferenceOptimization(args)
        
        assert po.algo == "Diffusion DPO"
        assert po.is_reward_based() is False
    
    def test_mapo_not_srpo(self):
        """Test that MaPO doesn't trigger SRPO"""
        args = Mock()
        args.mapo_beta = 1.0
        args.srpo_beta = None
        args.ddo_beta = None
        args.ddo_alpha = None
        args.bpo_beta = None
        args.bpo_lambda = None
        args.beta_dpo = None
        args.sdpo_beta = None
        args.simpo_beta = None
        args.cpo_beta = None
        
        po = PreferenceOptimization(args)
        
        assert po.algo == "MaPO"
        assert po.is_reward_based() is False


class TestSRPOArgsValidation:
    """Test SRPO args validation"""
    
    def test_missing_positive_prompt(self):
        """Test that missing positive_prompt is handled"""
        args = Mock()
        args.srpo_beta = 1.0
        args.srpo_positive_prompt = None  # Missing!
        args.srpo_negative_prompt = "Bad"

        # Set all other PO methods to None
        args.ddo_beta = None
        args.ddo_alpha = None
        args.bpo_beta = None
        args.bpo_lambda = None
        args.beta_dpo = None
        args.sdpo_beta = None
        args.mapo_beta = None
        args.simpo_beta = None
        args.cpo_beta = None

        # This should be caught by assert_po_variables
        # For now, just test that PreferenceOptimization can be created
        # The actual validation happens in assert_po_variables
        po = PreferenceOptimization(args)
        assert po.algo == "SRPO"
    
    def test_default_use_inversion_false(self):
        """Test that use_inversion defaults to False when not set"""
        # Use spec_set to make Mock raise AttributeError for undefined attributes
        from argparse import Namespace
        args = Namespace()
        args.srpo_beta = 1.0
        args.srpo_positive_prompt = "Good"
        args.srpo_negative_prompt = "Bad"
        # Don't set srpo_use_inversion - will use getattr default

        # Set all other PO methods to None
        args.ddo_beta = None
        args.ddo_alpha = None
        args.bpo_beta = None
        args.bpo_lambda = None
        args.beta_dpo = None
        args.sdpo_beta = None
        args.mapo_beta = None
        args.simpo_beta = None
        args.cpo_beta = None

        po = PreferenceOptimization(args)
        assert po.args["use_inversion"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
