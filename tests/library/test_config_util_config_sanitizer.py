import argparse

import pytest
from voluptuous import error

from library.config_util import ConfigSanitizer, merge_dicts, validate_and_convert_scalar_or_twodim, validate_and_convert_twodim


def test_init_requires_one_mode():
    """Test that at least one mode must be enabled."""
    with pytest.raises(AssertionError):
        ConfigSanitizer(False, False, False, False)

    # These should not raise errors
    ConfigSanitizer(True, False, False, False)
    ConfigSanitizer(False, True, False, False)
    ConfigSanitizer(False, False, True, False)
    ConfigSanitizer(True, True, True, True)


def test_merge_dict():
    """Test the static _merge_dict function."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    dict3 = {"a": 5}  # This should override dict1's "a" value
    # sanitizer = ConfigSanitizer(True, False, False, False)
    merged = merge_dicts(dict1, dict2, dict3)

    assert merged == {"a": 5, "b": 2, "c": 3, "d": 4}


def test_validate_and_convert_twodim():
    """Test the validate_and_convert_twodim function."""
    # Valid case
    result = validate_and_convert_twodim(float, [0.1, 0.2])
    assert result == (0.1, 0.2)

    # Invalid cases
    with pytest.raises(error.MultipleInvalid):
        validate_and_convert_twodim(float, [0.1, 0.2, 0.3])  # Too many elements

    with pytest.raises(error.MultipleInvalid):
        validate_and_convert_twodim(float, ["not", "floats"])  # Wrong type


def test_validate_and_convert_scalar_or_twodim():
    """Test the validate_and_convert_scalar_or_twodim function."""
    # Test scalar conversion
    result = validate_and_convert_scalar_or_twodim(float, 0.5)
    assert result == (0.5, 0.5)

    # Test two-dimensional conversion
    result = validate_and_convert_scalar_or_twodim(float, [0.1, 0.2])
    assert result == (0.1, 0.2)

    # Invalid cases
    with pytest.raises(error.MultipleInvalid):
        validate_and_convert_scalar_or_twodim(float, [0.1, 0.2, 0.3])  # Too many elements

    with pytest.raises(error.MultipleInvalid):
        validate_and_convert_scalar_or_twodim(float, "not a float or list")  # Wrong type


def test_dreambooth_schema():
    """Test the schema for DreamBooth mode."""
    sanitizer = ConfigSanitizer(True, False, False, False)

    valid_config = {
        "general": {
            "resolution": 512,
            "batch_size": 1,
        },
        "datasets": [
            {
                "resolution": [768, 768],
                "batch_size": 2,
                "subsets": [{"image_dir": "/path/to/images", "class_tokens": "person", "caption_extension": ".txt"}],
            }
        ],
    }

    # This should not raise an error
    result = sanitizer.sanitize_user_config(valid_config)
    assert result is not None

    # Test with invalid configuration (missing required field)
    invalid_config = {
        "general": {
            "resolution": 512,
        },
        "datasets": [
            {
                "subsets": [
                    {
                        # Missing required 'image_dir'
                        "class_tokens": "person"
                    }
                ]
            }
        ],
    }

    with pytest.raises(error.MultipleInvalid):
        sanitizer.sanitize_user_config(invalid_config)


def test_finetuning_schema():
    """Test the schema for FineTuning mode."""
    sanitizer = ConfigSanitizer(False, True, False, False)

    valid_config = {
        "general": {
            "resolution": 512,
            "batch_size": 1,
        },
        "datasets": [
            {
                "resolution": [768, 768],
                "batch_size": 2,
                "subsets": [{"metadata_file": "/path/to/metadata.json", "image_dir": "/path/to/images"}],
            }
        ],
    }

    # This should not raise an error
    result = sanitizer.sanitize_user_config(valid_config)
    assert result is not None

    # Test with invalid configuration
    invalid_config = {
        "general": {
            "resolution": 512,
        },
        "datasets": [
            {
                "subsets": [
                    {
                        # Missing required 'metadata_file'
                        "image_dir": "/path/to/images"
                    }
                ]
            }
        ],
    }

    with pytest.raises(error.MultipleInvalid):
        sanitizer.sanitize_user_config(invalid_config)


def test_controlnet_schema():
    """Test the schema for ControlNet mode."""
    sanitizer = ConfigSanitizer(False, False, True, False)

    valid_config = {
        "general": {
            "resolution": 512,
            "batch_size": 1,
        },
        "datasets": [
            {
                "resolution": [768, 768],
                "batch_size": 2,
                "subsets": [
                    {"image_dir": "/path/to/images", "conditioning_data_dir": "/path/to/conditioning", "caption_extension": ".txt"}
                ],
            }
        ],
    }

    # This should not raise an error
    result = sanitizer.sanitize_user_config(valid_config)
    assert result is not None

    # Test with invalid configuration
    invalid_config = {
        "general": {
            "resolution": 512,
        },
        "datasets": [
            {
                "subsets": [
                    {
                        "image_dir": "/path/to/images",
                        # Missing required 'conditioning_data_dir'
                    }
                ]
            }
        ],
    }

    with pytest.raises(error.MultipleInvalid):
        sanitizer.sanitize_user_config(invalid_config)


def test_mixed_mode():
    """Test when multiple modes are enabled."""
    sanitizer = ConfigSanitizer(True, True, False, False)

    # Valid DreamBooth style config with mixed mode enabled
    db_config = {
        "general": {
            "resolution": 512,
        },
        "datasets": [{"subsets": [{"image_dir": "/path/to/images", "class_tokens": "person"}]}],
    }

    # Valid FineTuning style config with mixed mode enabled
    ft_config = {
        "general": {
            "resolution": 512,
        },
        "datasets": [{"subsets": [{"metadata_file": "/path/to/metadata.json", "image_dir": "/path/to/images"}]}],
    }

    # Invalid mixed config (mixing DB and FT subsets in same dataset)
    invalid_mixed = {
        "general": {
            "resolution": 512,
        },
        "datasets": [
            {
                "subsets": [
                    {"image_dir": "/path/to/images", "class_tokens": "person"},
                    {"metadata_file": "/path/to/metadata.json", "image_dir": "/path/to/images"},
                ]
            }
        ],
    }

    # These should not raise errors
    result_db = sanitizer.sanitize_user_config(db_config)
    assert result_db is not None

    result_ft = sanitizer.sanitize_user_config(ft_config)
    assert result_ft is not None

    # This should raise an error due to mixing subset types
    with pytest.raises(error.MultipleInvalid):
        sanitizer.sanitize_user_config(invalid_mixed)


def test_dropout_schema():
    """Test when dropout mode is enabled."""
    # Test with dropout enabled
    sanitizer_with_dropout = ConfigSanitizer(True, False, False, True)

    valid_config = {
        "general": {
            "resolution": 512,
            "caption_dropout_rate": 0.1,
            "caption_dropout_every_n_epochs": 2,
            "caption_tag_dropout_rate": 0.2,
        },
        "datasets": [{"subsets": [{"image_dir": "/path/to/images", "caption_dropout_rate": 0.3}]}],
    }

    # This should not raise an error with dropout enabled
    result = sanitizer_with_dropout.sanitize_user_config(valid_config)
    assert result is not None

    # Test the same config with dropout disabled
    sanitizer_without_dropout = ConfigSanitizer(True, False, False, False)

    # This should raise an error as dropout fields are not recognized
    with pytest.raises(error.MultipleInvalid):
        sanitizer_without_dropout.sanitize_user_config(valid_config)


def test_argparse_namespace():
    """Test the sanitization of argparse Namespace objects."""
    sanitizer = ConfigSanitizer(True, False, False, False)

    # Create a valid Namespace
    args = argparse.Namespace(
        resolution=512,
        batch_size=1,
        train_batch_size=2,  # This should be mapped to batch_size in validation
        debug_dataset=True,
        face_crop_aug_range=None,  # This is allowed to be None
    )

    # This should not raise an error
    result = sanitizer.sanitize_argparse_namespace(args)
    assert result is not None

    # Test with invalid value type
    invalid_args = argparse.Namespace(
        resolution="not an int",  # Should be int
        batch_size=1,
    )

    with pytest.raises(error.MultipleInvalid):
        sanitizer.sanitize_argparse_namespace(invalid_args)
