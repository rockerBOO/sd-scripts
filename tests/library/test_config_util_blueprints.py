from pathlib import Path
import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from library.config_util import (
    Blueprint,
    BlueprintGenerator,
    BucketDatasetParams,
    ConfigSanitizer,
    ControlNetDatasetParams,
    ControlNetSubsetParams,
    DatasetBlueprint,
    DatasetGroupBlueprint,
    DreamBoothDatasetParams,
    DreamBoothSubsetParams,
    FineTuningDatasetParams,
    FineTuningSubsetParams,
    SubsetBlueprint,
    generate_controlnet_subsets_config_by_subdirs,
    generate_dataset_group_by_blueprint,
    generate_dreambooth_subsets_config_by_subdirs,
)


@pytest.fixture
def config_sanitizer():
    sanitizer = ConfigSanitizer(True, True, True, True)
    sanitizer.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME = {"batch_size": "batch_size", "width": "width", "height": "height"}
    return sanitizer


@pytest.fixture
def blueprint_generator(config_sanitizer):
    return BlueprintGenerator(config_sanitizer)


def generate_dreambooth_dataset(train_dir: Path, n=3) -> Path:
    subdir = train_dir / "dreambooth_person_photo"
    subdir.mkdir()

    for i in range(n):
        (subdir / f"img{i}.jpg").touch()
        with open(subdir / f"img{i}.caption", "w") as f:
            f.write(f"caption{i}")

    return subdir


def generate_controlnet_dataset(train_dir: Path, n=3) -> tuple[Path, Path]:
    subdir = train_dir / "controlnet_person_photo"
    subdir.mkdir()

    for i in range(n):
        (subdir / f"img{i}.jpg").touch()
        with open(subdir / f"img{i}.caption", "w") as f:
            f.write(f"caption{i}")

    subdir_conditioning = train_dir / "controlnet_conditioning"
    subdir_conditioning.mkdir()

    for i in range(n):
        (subdir_conditioning / f"img{i}.jpg").touch()

    return subdir, subdir_conditioning


def generate_finetuning_dataset(train_dir: Path) -> Path:
    subdir = train_dir / "finetuning_person_photo"
    subdir.mkdir()

    (subdir / "img1.jpg").touch()
    with open(subdir / "img1.caption", "w") as f:
        f.write("caption1")

    with open(subdir / "metadata.jsonl", "w") as f:
        json.dump({str(subdir / "img1.jpg"): {"caption": "caption1"}}, f)

    return subdir


def test_search_value():
    """Test search_value method to ensure it finds values in the fallback dicts."""
    fallbacks = [{"a": 1}, {"b": 2}, {"c": 3, "a": 4}]

    # Should find first occurrence
    assert BlueprintGenerator.search_value("a", fallbacks) == 1

    # Should find in second dict
    assert BlueprintGenerator.search_value("b", fallbacks) == 2

    # Should return default for missing key
    assert BlueprintGenerator.search_value("d", fallbacks) == None
    assert BlueprintGenerator.search_value("d", fallbacks, "default") == "default"


def test_generate_dreambooth_dataset(blueprint_generator: BlueprintGenerator):
    """Test generate method creates DreamBooth dataset blueprint correctly."""

    # Create test configuration
    user_config = {
        "general": {"batch_size": 8, "resolution": (1024, 1024), "validation_split": 0.1},
        "datasets": [{"subsets": [{"image_dir": "/path/to/dreambooth", "is_reg": False, "class_tokens": "person photo"}]}],
    }

    args = argparse.Namespace(batch_size=4, width=1024, height=1024)

    # Call the method under test
    with patch.object(blueprint_generator.sanitizer, "sanitize_user_config", return_value=user_config):
        with patch.object(blueprint_generator.sanitizer, "sanitize_argparse_namespace", return_value=args):
            result = blueprint_generator.generate(user_config, args)

    assert isinstance(result, Blueprint)
    assert isinstance(result.dataset_group, DatasetGroupBlueprint)
    assert len(result.dataset_group.datasets) == 1
    assert result.dataset_group.datasets[0].is_controlnet is False
    assert result.dataset_group.datasets[0].is_dreambooth is True


def test_generate_controlnet_dataset(blueprint_generator: BlueprintGenerator):
    """Test generate method creates ControlNet dataset blueprint correctly."""

    # Create test configuration
    user_config = {
        "general": {"batch_size": 8, "width": 512, "height": 512},
        "datasets": [{"subsets": [{"image_dir": "/path/to/controlnet", "conditioning_data_dir": "/path/to/conditioning"}]}],
    }

    args = argparse.Namespace(batch_size=4, width=1024, height=1024)

    # Call the method under test
    with patch.object(blueprint_generator.sanitizer, "sanitize_user_config", return_value=user_config):
        with patch.object(blueprint_generator.sanitizer, "sanitize_argparse_namespace", return_value=args):
            result = blueprint_generator.generate(user_config, args)

    assert isinstance(result, Blueprint)
    assert isinstance(result.dataset_group, DatasetGroupBlueprint)
    assert len(result.dataset_group.datasets) == 1
    assert result.dataset_group.datasets[0].is_controlnet is True
    assert result.dataset_group.datasets[0].is_dreambooth is True


def test_generate_finetuning_dataset(blueprint_generator: BlueprintGenerator):
    """Test generate method creates FineTuning dataset blueprint correctly."""
    # Create test configuration
    user_config = {
        "general": {"batch_size": 8, "width": 512, "height": 512},
        "datasets": [{"subsets": [{"image_dir": "/path/to/finetuning", "metadata_file": "/path/to/metadata.json"}]}],
    }

    args = argparse.Namespace(batch_size=4, width=1024, height=1024)

    # Call the method under test
    with patch.object(blueprint_generator.sanitizer, "sanitize_user_config", return_value=user_config):
        with patch.object(blueprint_generator.sanitizer, "sanitize_argparse_namespace", return_value=args):
            result = blueprint_generator.generate(user_config, args)

    assert isinstance(result, Blueprint)
    assert isinstance(result.dataset_group, DatasetGroupBlueprint)
    assert len(result.dataset_group.datasets) == 1
    assert result.dataset_group.datasets[0].is_controlnet is False
    assert result.dataset_group.datasets[0].is_dreambooth is False


def test_generate_params_by_fallbacks():
    """Test generate_params_by_fallbacks method constructs parameters correctly."""
    # Setup test data
    fallbacks = [
        {"batch_size": 8},
        {"resize_interpolation": None},  # Fixed: was a set, now a dict
        {"batch_size": 4, "validation_split": 0.2},
    ]

    # Using the real implementation for this test
    BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}
    result = BlueprintGenerator.generate_params_by_fallbacks(FineTuningDatasetParams, fallbacks)

    # Assert the parameters were constructed with the right values
    assert result.batch_size == 8  # First occurrence in fallbacks
    assert result.validation_split == 0.2  # From third fallback

    # Additional assertions to check default values are maintained
    assert result.resolution == (512, 512)
    assert result.network_multiplier == 1.0
    assert result.debug_dataset is False
    assert result.validation_seed is None
    assert result.resize_interpolation is None


def test_generate_bucket_by_fallbacks():
    """Test generate_bucket_by_fallbacks method constructs bucket parameters correctly."""
    # Setup test data
    fallbacks = [
        {"min_bucket_reso": 128, "max_bucket_reso": 2048},
        {"bucket_reso_steps": 32, "min_bucket_reso": 256},
        {"bucket_no_upscale": True},
    ]

    # Using the real implementation for this test
    BlueprintGenerator.BLUEPRINT_BUCKET_NAME_TO_CONFIG_OPTNAME = {}
    result = BlueprintGenerator.generate_bucket_by_fallbacks(BucketDatasetParams, fallbacks)

    # Assert the parameters were constructed with the right values
    assert result.min_bucket_reso == 128  # First occurrence in fallbacks
    assert result.max_bucket_reso == 2048  # First occurrence in fallbacks
    assert result.bucket_reso_steps == 32  # From second fallback
    assert result.bucket_no_upscale is True  # From third fallback


# Tests for helper functions


def test_generate_dreambooth_subsets_config_by_subdirs(tmp_path):
    """Test generate_dreambooth_subsets_config_by_subdirs creates correct configurations."""
    # Create temporary directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create subdirectories with naming pattern: {num_repeats}_{class_tokens}
    subdir1 = train_dir / "10_person_photo"
    subdir1.mkdir()
    subdir2 = train_dir / "5_landscape"
    subdir2.mkdir()

    # Call the function
    result = generate_dreambooth_subsets_config_by_subdirs(train_dir, None)

    print(result)

    # Expected result
    expected = [
        {"image_dir": str(subdir2), "num_repeats": 5, "is_reg": False, "class_tokens": "landscape"},
        {"image_dir": str(subdir1), "num_repeats": 10, "is_reg": False, "class_tokens": "person_photo"},
    ]

    # Test equality excluding string representation of mocks
    assert len(result) == len(expected)
    for res, exp in zip(result, expected):
        assert res["num_repeats"] == exp["num_repeats"]
        assert res["is_reg"] == exp["is_reg"]
        assert res["class_tokens"] == exp["class_tokens"]


def test_generate_controlnet_subsets_config_by_subdirs(tmp_path):
    """Test generate_controlnet_subsets_config_by_subdirs creates correct configurations."""
    # Create temporary directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()

    # Call the function with the temporary directories
    result = generate_controlnet_subsets_config_by_subdirs(str(train_dir), str(conditioning_dir), ".json")

    # Expected result
    expected = [
        {
            "image_dir": str(train_dir),
            "conditioning_data_dir": str(conditioning_dir),
            "caption_extension": ".json",
            "num_repeats": 1,
        }
    ]

    # Test equality
    assert result == expected


# Mock the strategy classes
class MockTokenizeStrategy:
    @classmethod
    def get_strategy(cls):
        return MagicMock()


class MockTextEncoderOutputsCachingStrategy:
    @classmethod
    def get_strategy(cls):
        return MagicMock()


class MockLatentsCachingStrategy:
    @classmethod
    def get_strategy(cls):
        return MagicMock()


class MockTextEncodingStrategy:
    @classmethod
    def get_strategy(cls):
        return MagicMock()


# Test the function with various cases
def test_generate_dataset_group_by_blueprint_basic(tmp_path):
    """Test basic dataset group generation without validation split."""

    # Create temporary directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    subdir1 = generate_dreambooth_dataset(train_dir)

    # Mock the dataset classes
    with (
        patch("library.train_util.TokenizeStrategy", MockTokenizeStrategy),
        patch("library.train_util.TextEncoderOutputsCachingStrategy", MockTextEncoderOutputsCachingStrategy),
        patch("library.train_util.LatentsCachingStrategy", MockLatentsCachingStrategy),
        patch("library.train_util.TextEncodingStrategy", MockTextEncodingStrategy),
        patch("library.train_util.DreamBoothSubset"),
        patch("library.train_util.DreamBoothDataset"),
        patch("library.train_util.FineTuningSubset"),
        patch("library.train_util.FineTuningDataset"),
        patch("library.train_util.ControlNetSubset"),
        patch("library.train_util.ControlNetDataset"),
        patch("library.train_util.DatasetGroup") as mock_dataset_group,
    ):
        # Create a simple dataset blueprint
        subset_blueprint = SubsetBlueprint(
            params=DreamBoothSubsetParams(
                image_dir=subdir1,
            )
        )

        dataset_blueprint = DatasetBlueprint(
            is_dreambooth=True,
            is_controlnet=False,
            bucket=BucketDatasetParams(),
            params=DreamBoothDatasetParams(batch_size=1, resolution=(1024, 1024), validation_split=0.0),
            subsets=[subset_blueprint],
        )

        dataset_group_blueprint = DatasetGroupBlueprint(datasets=[dataset_blueprint])

        # Call the function
        train, val = generate_dataset_group_by_blueprint(dataset_group_blueprint)

        assert train.num_train_images == 3
        assert train.num_reg_images == 0
        assert len(train.image_data.keys()) == 3

        assert val is None


def test_generate_dataset_group_by_blueprint_with_validation(tmp_path):
    """Test dataset group generation with validation split."""

    # Create temporary directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    subdir1 = generate_dreambooth_dataset(train_dir, n=5)

    # Mock the dataset classes
    with (
        patch("library.train_util.TokenizeStrategy", MockTokenizeStrategy),
        patch("library.train_util.TextEncoderOutputsCachingStrategy", MockTextEncoderOutputsCachingStrategy),
        patch("library.train_util.LatentsCachingStrategy", MockLatentsCachingStrategy),
        patch("library.train_util.TextEncodingStrategy", MockTextEncodingStrategy),
        patch("library.train_util.DreamBoothSubset"),
        patch("library.train_util.DreamBoothDataset"),
        patch("library.train_util.FineTuningSubset"),
        patch("library.train_util.FineTuningDataset"),
        patch("library.train_util.ControlNetSubset"),
        patch("library.train_util.ControlNetDataset"),
    ):
        # Create a dataset blueprint with validation split
        subset_blueprint = SubsetBlueprint(
            params=DreamBoothSubsetParams(
                image_dir=subdir1,
            )
        )

        dataset_blueprint = DatasetBlueprint(
            is_dreambooth=True,
            is_controlnet=False,
            bucket=BucketDatasetParams(),
            params=DreamBoothDatasetParams(
                batch_size=1,
                resolution=(1024, 1024),
                resize_interpolation="bicubic",
                validation_split=0.2,  # 20% validation split
            ),
            subsets=[subset_blueprint],
        )

        dataset_group_blueprint = DatasetGroupBlueprint(datasets=[dataset_blueprint])

        # Call the function
        train, val = generate_dataset_group_by_blueprint(dataset_group_blueprint)

        assert train.num_train_images == 4
        assert train.num_reg_images == 0
        assert len(train.image_data.keys()) == 4

        assert val is not None
        assert val.num_train_images == 1
        assert val.num_reg_images == 0
        assert len(val.image_data.keys()) == 1


def test_generate_dataset_group_by_blueprint_multiple_dataset_types(tmp_path):
    """Test generation with multiple dataset types (DreamBooth, ControlNet, FineTuning)."""

    # Create temporary directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    subdir1 = generate_dreambooth_dataset(train_dir)
    subdir2, subdir2_conditioning = generate_controlnet_dataset(train_dir)
    subdir3 = generate_finetuning_dataset(train_dir)

    # Mock the dataset classes
    with (
        patch("library.config_util.logger.warning"),
        patch("library.train_util.TokenizeStrategy", MockTokenizeStrategy),
        patch("library.train_util.TextEncoderOutputsCachingStrategy", MockTextEncoderOutputsCachingStrategy),
        patch("library.train_util.LatentsCachingStrategy", MockLatentsCachingStrategy),
        patch("library.train_util.TextEncodingStrategy", MockTextEncodingStrategy),
    ):
        dreambooth_subset_blueprint = SubsetBlueprint(
            params=DreamBoothSubsetParams(
                image_dir=subdir1,
            )
        )

        # Create three different dataset blueprints
        dreambooth_blueprint = DatasetBlueprint(
            bucket=BucketDatasetParams(),
            params=DreamBoothDatasetParams(resolution=(1024, 1024), batch_size=1, validation_split=0.0),
            subsets=[dreambooth_subset_blueprint],
            is_dreambooth=True,
            is_controlnet=False,
        )

        controlnet_subset_blueprint = SubsetBlueprint(
            params=ControlNetSubsetParams(image_dir=subdir2, conditioning_data_dir=subdir2_conditioning)
        )

        controlnet_blueprint = DatasetBlueprint(
            bucket=BucketDatasetParams(),
            params=ControlNetDatasetParams(batch_size=1, resolution=(1024, 1024), validation_split=0.0),
            subsets=[controlnet_subset_blueprint],
            is_dreambooth=True,
            is_controlnet=True,
        )

        finetuning_subset_blueprint = SubsetBlueprint(
            params=FineTuningSubsetParams(
                image_dir=str(subdir3),
                metadata_file=str(subdir3 / "metadata.jsonl"),
            )
        )

        finetuning_blueprint = DatasetBlueprint(
            bucket=BucketDatasetParams(),
            params=FineTuningDatasetParams(batch_size=1, resolution=(1024, 1024), validation_split=0.0),
            subsets=[finetuning_subset_blueprint],
            is_dreambooth=False,
            is_controlnet=False,
        )

        dataset_group_blueprint = DatasetGroupBlueprint(datasets=[dreambooth_blueprint, controlnet_blueprint, finetuning_blueprint])

        # Call the function
        train, val = generate_dataset_group_by_blueprint(dataset_group_blueprint)

        assert train.num_train_images == 7
        assert train.num_reg_images == 0
        assert len(train.image_data.keys()) == 7

        assert val is None


def test_generate_dataset_group_by_blueprint_invalid_validation_split(tmp_path):
    """Test handling of invalid validation split values."""

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    subdir1 = generate_dreambooth_dataset(train_dir)

    # Mock the dataset classes and logger
    with (
        patch("library.config_util.logger.warning"),
        patch("library.train_util.TokenizeStrategy", MockTokenizeStrategy),
        patch("library.train_util.TextEncoderOutputsCachingStrategy", MockTextEncoderOutputsCachingStrategy),
        patch("library.train_util.LatentsCachingStrategy", MockLatentsCachingStrategy),
        patch("library.train_util.TextEncodingStrategy", MockTextEncodingStrategy),
        patch("library.train_util.FineTuningSubset"),
        patch("library.train_util.FineTuningDataset"),
        patch("library.train_util.DatasetGroup"),
    ):
        # Create a dataset blueprint with invalid validation split
        subset_blueprint = SubsetBlueprint(
            params=DreamBoothSubsetParams(
                image_dir=subdir1,
            )
        )

        # Case 1: validation_split < 0.0
        dataset_blueprint_neg = DatasetBlueprint(
            is_controlnet=False,
            is_dreambooth=True,
            bucket=BucketDatasetParams(),
            params=DreamBoothDatasetParams(
                batch_size=1,
                resolution=(1024, 1024),
                validation_split=-0.1,  # Invalid negative value
            ),
            subsets=[subset_blueprint],
        )

        # Case 2: validation_split > 1.0
        dataset_blueprint_over = DatasetBlueprint(
            is_controlnet=False,
            is_dreambooth=True,
            bucket=BucketDatasetParams(),
            params=DreamBoothDatasetParams(
                batch_size=1,
                resolution=(1024, 1024),
                validation_split=1.5,  # Invalid value over 1.0
            ),
            subsets=[subset_blueprint],
        )

        # Test with negative validation split
        dataset_group_blueprint = DatasetGroupBlueprint(datasets=[dataset_blueprint_neg])
        _, val = generate_dataset_group_by_blueprint(dataset_group_blueprint)

        assert val is None

        # Test with validation split > 1.0
        dataset_group_blueprint = DatasetGroupBlueprint(datasets=[dataset_blueprint_over])
        _, val = generate_dataset_group_by_blueprint(dataset_group_blueprint)

        assert val is None
