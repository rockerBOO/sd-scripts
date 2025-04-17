from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from dataclasses import asdict
import dataclasses
from torch import Tensor
from tqdm import tqdm
from unittest.mock import Mock, MagicMock, patch
import logging

from library.config_util import (
    BlueprintGenerator,
    BucketDatasetParams,
    ConfigSanitizer,
    DatasetBlueprint,
    DatasetGroupBlueprint,
    DreamBoothDatasetParams,
    DreamBoothSubsetParams,
    SubsetBlueprint,
    DatasetGroup,
    generate_dataset_group_by_blueprint,
)
from library.train_util import (
    DreamBoothSubset,
    ImageInfo,
    ImageSet,
    BucketDataset,
    MultiBucketDataset,
    BucketManager,
)


def generate_dreambooth_dataset(train_dir: Path, n=3) -> Path:
    subdir = train_dir / "dreambooth_person_photo"
    subdir.mkdir()

    for i in range(n):
        # Create a simple image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = subdir / f"img{i}.jpg"
        img.save(img_path)

        (subdir / f"img{i}.jpg").touch()
        with open(subdir / f"img{i}.caption", "w") as f:
            f.write(f"caption{i}")

    return subdir


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


def test_basic_initialization():
    """Test that ImageInfo initializes with the correct attributes."""
    img_info = ImageInfo(
        image_key="test_image",
        num_repeats=2,
        caption="A test image",
        is_reg=False,
        absolute_path="/path/to/image.jpg",
        role="image",
    )

    assert img_info.image_key == "test_image"
    assert img_info.num_repeats == 2
    assert img_info.caption == "A test image"
    assert img_info.is_reg is False
    assert img_info.absolute_path == "/path/to/image.jpg"
    assert img_info.role == "image"  # Default value
    assert img_info.image_size is None
    assert img_info.latents is None


def test_set_creation():
    """Test creating different types of image sets."""
    # Create test ImageInfo objects
    img1 = ImageInfo("img1", 1, "Image 1", False, "/path/1.jpg", "image")
    img2 = ImageInfo("img2", 1, "Image 2", False, "/path/2.jpg", "image")

    # Test single image set
    single_set = ImageSet.create_single(img1)
    assert single_set.get_image_count() == 1
    assert single_set.get_image_by_index(0) == img1
    assert single_set.set_type == "single"
    assert not single_set.is_preference_pair

    # Test preference pair
    pref_set = ImageSet.create_preference_pair(img1, img2)
    assert pref_set.get_image_count() == 2
    assert pref_set.get_image_by_role("preferred") == img1
    assert pref_set.get_image_by_role("non_preferred") == img2
    assert pref_set.is_preference_pair

    # Test image+mask pair
    mask_set = ImageSet.create_image_mask_pair(img1, img2)
    assert mask_set.get_image_count() == 2
    assert mask_set.get_image_by_role("image") == img1
    assert mask_set.get_image_by_role("mask") == img2
    assert mask_set.is_image_mask_pair


def test_bucket_creation():
    """Test that BucketManager correctly creates buckets."""
    # Create bucket manager with upscaling allowed
    manager = BucketManager(no_upscale=False, max_reso=(1024, 1024), min_size=256, max_size=1024, reso_steps=64)

    # Make buckets should create a set of predefined resolutions
    manager.make_buckets()

    # Verify buckets were created
    assert len(manager.predefined_resos) > 0
    assert len(manager.buckets) == len(manager.resos)
    assert all(isinstance(bucket, list) for bucket in manager.buckets)


def test_image_assignment():
    """Test that images are assigned to appropriate buckets."""
    # Create bucket manager
    manager = BucketManager(no_upscale=False, max_reso=(512, 512), min_size=256, max_size=512, reso_steps=64)
    manager.make_buckets()

    # Test a few image sizes
    reso1, size1, error1 = manager.select_bucket(400, 300)
    reso2, size2, error2 = manager.select_bucket(800, 600)

    # Verify results
    assert reso1 in manager.resos
    assert reso2 in manager.resos

    # Add images to buckets
    manager.add_image(reso1, "image1")
    manager.add_image(reso2, "image2")

    # Check that images were added to correct buckets
    bucket1_id = manager.reso_to_id[reso1]
    bucket2_id = manager.reso_to_id[reso2]

    assert "image1" in manager.buckets[bucket1_id]
    assert "image2" in manager.buckets[bucket2_id]


def test_no_upscale_mode():
    """Test that no_upscale mode works correctly."""
    # Create bucket manager with no upscaling
    manager = BucketManager(no_upscale=True, max_reso=(512, 512), min_size=256, max_size=512, reso_steps=64)

    # Test with an image smaller than max_area
    small_img_width, small_img_height = 300, 200
    reso, size, _ = manager.select_bucket(small_img_width, small_img_height)

    # In no_upscale mode, the resolution should be the original or smaller
    assert reso[0] <= small_img_width
    assert reso[1] <= small_img_height
    assert reso[0] % manager.reso_steps == 0
    assert reso[1] % manager.reso_steps == 0


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing BucketDataset."""
    dataset = MagicMock()
    dataset.image_data = {
        "img1": MagicMock(image_size=(512, 384), num_repeats=2, bucket_reso=None, resized_size=None),
        "img2": MagicMock(image_size=(768, 512), num_repeats=1, bucket_reso=None, resized_size=None),
    }
    dataset.width = 512
    dataset.height = 512
    dataset.debug_dataset = False
    return dataset


@patch("random.seed")
@patch("random.shuffle")
def test_shuffle_buckets(mock_shuffle, mock_seed, mock_dataset):
    """Test that shuffling happens correctly."""
    # Create and prepare bucket dataset
    bucket_ds = BucketDataset(
        base_dataset=mock_dataset, batch_size=1, min_bucket_reso=256, max_bucket_reso=1024, bucket_reso_steps=64
    )
    bucket_ds.bucket_manager = MagicMock()
    bucket_ds.buckets_indices = [1, 2, 3]

    # Set epoch and seed
    mock_dataset.seed = 42
    mock_dataset.current_epoch = 1

    # Shuffle
    bucket_ds.shuffle_buckets()

    # Verify calls
    mock_seed.assert_called_once_with(42 + 1)
    mock_shuffle.assert_called()
    bucket_ds.bucket_manager.shuffle.assert_called_once()


@pytest.fixture
def default_params():
    """Fixture providing default parameters for DreamBoothSubset initialization."""
    return DreamBoothSubsetParams(image_dir="/path/to/images", is_reg=False, class_tokens="person")


@pytest.fixture
def dreambooth_subset(default_params):
    """Fixture providing a standard DreamBoothSubset instance."""
    return DreamBoothSubset(**asdict(default_params))


@pytest.fixture
def preference_params(default_params: DreamBoothSubsetParams) -> DreamBoothSubsetParams:
    """Fixture providing parameters for preference-enabled DreamBoothSubset."""
    params = dataclasses.replace(
        default_params,
        **{
            "preference": True,
            "preference_caption_prefix": "good:",
            "preference_caption_suffix": "(better)",
            "non_preference_caption_prefix": "bad:",
            "non_preference_caption_suffix": "(worse)",
        },
    )
    return params


@pytest.fixture
def preference_dreambooth_subset(preference_params: DreamBoothSubsetParams):
    """Fixture providing a preference-enabled DreamBoothSubset instance."""
    return DreamBoothSubset(**asdict(preference_params))


@pytest.fixture
def temp_image_text_files(tmp_path):
    """
    Creates n pairs of temporary image.jpg and image.txt files in a temporary directory.

    Usage:
        @pytest.mark.parametrize("file_count", [1, 3, 5])
        def test_something(temp_image_text_files, file_count):
            file_paths = temp_image_text_files(file_count)
            # file_paths contains lists of jpg and txt paths
    """

    def _create_files(n=1):
        jpg_files = []
        txt_files = []

        for i in range(n):
            # Create image file
            jpg_path = tmp_path / f"image_{i}.jpg"
            jpg_path.write_bytes(b"dummy image content")
            jpg_files.append(jpg_path)

            # Create text file
            txt_path = tmp_path / f"image_{i}.txt"
            txt_path.write_text(f"Content for image_{i}")
            txt_files.append(txt_path)

        return {"jpg_files": jpg_files, "txt_files": txt_files, "dir": tmp_path}

    return _create_files


def test_init_basic(dreambooth_subset, default_params):
    """Test basic initialization of DreamBoothSubset."""
    assert dreambooth_subset.image_dir == default_params.image_dir
    assert dreambooth_subset.is_reg == default_params.is_reg
    assert dreambooth_subset.class_tokens == default_params.class_tokens
    assert dreambooth_subset.caption_extension == default_params.caption_extension
    assert dreambooth_subset.cache_info == default_params.cache_info


def test_caption_extension_format():
    """Test caption_extension formatting with and without dots."""
    # With dot in the extension
    params = DreamBoothSubsetParams(image_dir="/path/to/images", is_reg=False, class_tokens="person", caption_extension=".txt")
    subset_with_dot = DreamBoothSubset(**asdict(params))

    subset_with_dot = DreamBoothSubset(**asdict(params))
    assert subset_with_dot.caption_extension == ".txt"

    params = DreamBoothSubsetParams(image_dir="/path/to/images", is_reg=False, class_tokens="person", caption_extension="txt")
    # Without dot in the extension
    subset_without_dot = DreamBoothSubset(**asdict(params))
    assert subset_without_dot.caption_extension == ".txt"


def test_equality():
    """Test the equality method of DreamBoothSubset."""
    params1 = DreamBoothSubsetParams(image_dir="/path/to/images", is_reg=False, class_tokens="person")
    params2 = DreamBoothSubsetParams(image_dir="/path/to/images", is_reg=True, class_tokens="person")
    params2.image_dir = "/different/path"

    subset1 = DreamBoothSubset(**vars(params1))
    subset2 = DreamBoothSubset(**vars(params1))  # Same parameters
    subset3 = DreamBoothSubset(**vars(params2))  # Different image_dir

    assert subset1 == subset2
    assert subset1 != subset3
    assert subset1 != "not a subset"  # Test with non-DreamBoothSubset object


def test_multi_dataset_initialization():
    """Test initializing MultiBucketDataset with multiple datasets."""
    # Create mock datasets
    dataset1 = MagicMock()
    dataset1.width = 512
    dataset1.height = 512

    dataset2 = MagicMock()
    dataset2.width = 512
    dataset2.height = 512

    # Initialize multi-bucket dataset
    multi_ds = MultiBucketDataset(
        dataset_groups=[dataset1, dataset2], batch_size=1, min_bucket_reso=256, max_bucket_reso=1024, bucket_reso_steps=64
    )

    # Check initialization
    assert len(multi_ds.dataset_groups) == 2
    assert multi_ds.dataset_map == {}


def test_sync_state_to_datasets():
    """Test that state is synchronized to all datasets."""
    # Create mock datasets
    dataset1 = MagicMock()
    dataset2 = MagicMock()

    # Create multi-bucket dataset
    multi_ds = MultiBucketDataset(
        dataset_groups=[dataset1, dataset2],
        batch_size=1,
        min_bucket_reso=256,
        max_bucket_reso=1024,
    )

    # Set state
    multi_ds.set_current_epoch(5)
    multi_ds.set_current_step(100)
    multi_ds.set_max_train_steps(1000)
    multi_ds.set_seed(42)

    # Verify state was set on all datasets
    dataset1.set_current_epoch.assert_called_once_with(5)
    dataset2.set_current_epoch.assert_called_once_with(5)

    dataset1.set_current_step.assert_called_once_with(100)
    dataset2.set_current_step.assert_called_once_with(100)

    dataset1.set_max_train_steps.assert_called_once_with(1000)
    dataset2.set_max_train_steps.assert_called_once_with(1000)

    dataset1.set_seed.assert_called_once_with(42)
    dataset2.set_seed.assert_called_once_with(42)


def test_multi_dataset_datasets(tmp_path):
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

        dataset = MultiBucketDataset(
            [train, train],
            batch_size=1,
            min_bucket_reso=256,
            max_bucket_reso=1024,
        )

        dataset.prepare_datasets()
        dataset.prepare_buckets()

        for item in dataset:
            assert isinstance(item["images"], Tensor)
            assert len(item["crop_top_lefts"]) == 2
            assert len(item["original_sizes"]) == 2
            assert len(item["target_sizes"]) == 2


# Create a fixture for the test data
@pytest.fixture
def mock_dataset_groups():
    # Create mock dataset groups
    dataset1 = Mock(spec=DatasetGroup)
    dataset2 = Mock(spec=DatasetGroup)

    # Set up image data for dataset1
    dataset1.image_data = {"img1": Mock(image_size=(800, 600), num_repeats=1), "img2": Mock(image_size=(1024, 768), num_repeats=2)}

    # Set up image data for dataset2
    dataset2.image_data = {"img3": Mock(image_size=(512, 512), num_repeats=1), "img4": Mock(image_size=(1280, 720), num_repeats=3)}

    # Set up get_resolutions method
    dataset1.get_resolutions.return_value = [(800, 600), (1024, 768)]
    dataset2.get_resolutions.return_value = [(512, 512), (1280, 720)]

    return [dataset1, dataset2]


# Test the prepare_buckets method
def test_prepare_buckets(mock_dataset_groups, caplog):
    # Set up logging
    caplog.set_level(logging.INFO)

    # Create MultiBucketDataset instance
    multi_bucket_dataset = MultiBucketDataset(
        dataset_groups=mock_dataset_groups, batch_size=2, min_bucket_reso=256, max_bucket_reso=1584, bucket_reso_steps=64
    )

    # Call prepare_buckets
    multi_bucket_dataset.prepare_buckets()

    # Verify bucket manager was created
    assert multi_bucket_dataset.bucket_manager is not None

    # Verify parameters were passed correctly to BucketManager
    assert multi_bucket_dataset.bucket_manager.min_size == 256
    assert multi_bucket_dataset.bucket_manager.max_size == 1584
    assert multi_bucket_dataset.bucket_manager.reso_steps == 64

    # Verify dataset_map was populated correctly
    assert len(multi_bucket_dataset.dataset_map) == 4
    assert "0_img1" in multi_bucket_dataset.dataset_map
    assert "0_img2" in multi_bucket_dataset.dataset_map
    assert "1_img3" in multi_bucket_dataset.dataset_map
    assert "1_img4" in multi_bucket_dataset.dataset_map

    # Verify buckets_indices was populated
    assert len(multi_bucket_dataset.buckets_indices) > 0

    # Verify bucket_info was populated
    assert "buckets" in multi_bucket_dataset.bucket_info

    # Check that images were added to buckets
    total_images = 0
    for bucket in multi_bucket_dataset.bucket_manager.buckets:
        total_images += len(bucket)

    # Total should be 7 (1 + 2 + 1 + 3 from the repeats)
    assert total_images == 7

    print(caplog.text)

    # Verify logging output
    assert "Assigning images to buckets from multiple datasets" in caplog.text
    assert "Number of images per bucket (including repeats)" in caplog.text
    assert "Mean aspect ratio error" in caplog.text


# Test with bucket_no_upscale=True
def test_prepare_buckets_no_upscale(mock_dataset_groups, caplog):
    # Set up logging
    caplog.set_level(logging.INFO)

    # Create MultiBucketDataset instance with bucket_no_upscale=True
    multi_bucket_dataset = MultiBucketDataset(
        dataset_groups=mock_dataset_groups,
        batch_size=2,
        min_bucket_reso=256,
        max_bucket_reso=1584,
        bucket_no_upscale=True,
        bucket_reso_steps=64,
    )

    # Call prepare_buckets
    multi_bucket_dataset.prepare_buckets()

    # Verify bucket manager was created with no_upscale=True
    assert multi_bucket_dataset.bucket_manager is not None
    assert multi_bucket_dataset.bucket_manager.no_upscale is True

    # Verify warning was logged
    assert "bucket_no_upscale is set - min/max bucket resolution ignored" in caplog.text
