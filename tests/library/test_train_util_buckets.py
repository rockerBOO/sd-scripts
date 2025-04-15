import argparse
import pytest
from tqdm import tqdm
from unittest.mock import MagicMock, patch

# Import the classes to test
# Assume these are in a module called 'bucket_system'
from library.config_util import BlueprintGenerator, ConfigSanitizer, DreamBoothSubsetParams, generate_dataset_group_by_blueprint
from library.train_util import (
    DreamBoothDataset,
    DreamBoothSubset,
    ImageInfo,
    ImageSetInfo,
    ImageSet,
    BucketBatchIndex,
    BucketDataset,
    MultiBucketDataset,
    BucketManager,
)


class TestImageInfo:
    def test_basic_initialization(self):
        """Test that ImageInfo initializes with the correct attributes."""
        img_info = ImageInfo(
            image_key="test_image", num_repeats=2, caption="A test image", is_reg=False, absolute_path="/path/to/image.jpg"
        )

        assert img_info.image_key == "test_image"
        assert img_info.num_repeats == 2
        assert img_info.caption == "A test image"
        assert img_info.is_reg is False
        assert img_info.absolute_path == "/path/to/image.jpg"
        assert img_info.role == "image"  # Default value
        assert img_info.image_size is None
        assert img_info.latents is None


class TestImageSet:
    def test_set_creation(self):
        """Test creating different types of image sets."""
        # Create test ImageInfo objects
        img1 = ImageInfo("img1", 1, "Image 1", False, "/path/1.jpg")
        img2 = ImageInfo("img2", 1, "Image 2", False, "/path/2.jpg")

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


class TestBucketManager:
    def test_bucket_creation(self):
        """Test that BucketManager correctly creates buckets."""
        # Create bucket manager with upscaling allowed
        manager = BucketManager(no_upscale=False, max_reso=(1024, 1024), min_size=256, max_size=1024, reso_steps=64)

        # Make buckets should create a set of predefined resolutions
        manager.make_buckets()

        # Verify buckets were created
        assert len(manager.predefined_resos) > 0
        assert len(manager.buckets) == len(manager.resos)
        assert all(isinstance(bucket, list) for bucket in manager.buckets)

    def test_image_assignment(self):
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

    def test_no_upscale_mode(self):
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


class TestBucketDataset:
    def test_prepare_buckets(self, mock_dataset):
        """Test bucket preparation in BucketDataset."""
        # Create bucket dataset
        bucket_ds = BucketDataset(
            base_dataset=mock_dataset, batch_size=1, min_bucket_reso=256, max_bucket_reso=1024, bucket_reso_steps=64
        )

        # Prepare buckets
        with patch("tqdm.tqdm", lambda x: x):  # Mock tqdm to avoid progress bar in tests
            bucket_ds.prepare_buckets()

        # Check that buckets were created
        assert bucket_ds.bucket_manager is not None
        assert len(bucket_ds.buckets_indices) > 0

        # Check that dataset length was set
        assert len(bucket_ds) == len(bucket_ds.buckets_indices)

    @patch("random.seed")
    @patch("random.shuffle")
    def test_shuffle_buckets(self, mock_shuffle, mock_seed, mock_dataset):
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
    return {
        "image_dir": "/path/to/images",
        "is_reg": False,
        "class_tokens": "person",
        "caption_extension": "txt",
        "cache_info": True,
        "alpha_mask": False,
        "num_repeats": 1,
        "shuffle_caption": True,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": ",",
        "secondary_separator": ";",
        "enable_wildcard": False,
        "color_aug": False,
        "flip_aug": False,
        "face_crop_aug_range": None,
        "random_crop": False,
        "caption_dropout_rate": 0.0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0.0,
        "caption_prefix": "",
        "caption_suffix": "",
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    }


@pytest.fixture
def dreambooth_subset(default_params):
    """Fixture providing a standard DreamBoothSubset instance."""
    return DreamBoothSubset(**default_params)


@pytest.fixture
def preference_params(default_params):
    """Fixture providing parameters for preference-enabled DreamBoothSubset."""
    params = default_params.copy()
    params.update(
        {
            "preference": True,
            "preference_caption_prefix": "good:",
            "preference_caption_suffix": "(better)",
            "non_preference_caption_prefix": "bad:",
            "non_preference_caption_suffix": "(worse)",
        }
    )
    return params


@pytest.fixture
def preference_dreambooth_subset(preference_params):
    """Fixture providing a preference-enabled DreamBoothSubset instance."""
    return DreamBoothSubset(**preference_params)


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
    assert dreambooth_subset.image_dir == default_params["image_dir"]
    assert dreambooth_subset.is_reg == default_params["is_reg"]
    assert dreambooth_subset.class_tokens == default_params["class_tokens"]
    assert dreambooth_subset.caption_extension == "." + default_params["caption_extension"]
    assert dreambooth_subset.cache_info == default_params["cache_info"]
    assert dreambooth_subset.preference_handler is None


def test_init_with_preference(preference_dreambooth_subset, preference_params):
    """Test initialization with preference handlers."""
    assert preference_dreambooth_subset.preference == preference_params["preference"]
    assert preference_dreambooth_subset.preference_handler is not None
    assert preference_dreambooth_subset.preference_handler.caption_prefix == preference_params["preference_caption_prefix"]
    assert preference_dreambooth_subset.preference_handler.caption_suffix == preference_params["preference_caption_suffix"]
    assert (
        preference_dreambooth_subset.preference_handler.non_preference_caption_prefix
        == preference_params["non_preference_caption_prefix"]
    )
    assert (
        preference_dreambooth_subset.preference_handler.non_preference_caption_suffix
        == preference_params["non_preference_caption_suffix"]
    )


def test_caption_extension_format():
    """Test caption_extension formatting with and without dots."""
    # With dot in the extension
    params = {
        "image_dir": "/path/to/images",
        "is_reg": False,
        "class_tokens": "person",
        "caption_extension": ".txt",  # Already has a dot
        "cache_info": True,
        "alpha_mask": False,
        "num_repeats": 1,
        "shuffle_caption": True,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": ",",
        "secondary_separator": ";",
        "enable_wildcard": False,
        "color_aug": False,
        "flip_aug": False,
        "face_crop_aug_range": None,
        "random_crop": False,
        "caption_dropout_rate": 0.0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0.0,
        "caption_prefix": "",
        "caption_suffix": "",
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    }

    subset_with_dot = DreamBoothSubset(**params)
    assert subset_with_dot.caption_extension == ".txt"

    # Without dot in the extension
    params["caption_extension"] = "txt"
    subset_without_dot = DreamBoothSubset(**params)
    assert subset_without_dot.caption_extension == ".txt"


def test_equality():
    """Test the equality method of DreamBoothSubset."""
    params1 = {
        "image_dir": "/path/to/images",
        "is_reg": False,
        "class_tokens": "person",
        "caption_extension": "txt",
        "cache_info": True,
        "alpha_mask": False,
        "num_repeats": 1,
        "shuffle_caption": True,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": ",",
        "secondary_separator": ";",
        "enable_wildcard": False,
        "color_aug": False,
        "flip_aug": False,
        "face_crop_aug_range": None,
        "random_crop": False,
        "caption_dropout_rate": 0.0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0.0,
        "caption_prefix": "",
        "caption_suffix": "",
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    }

    params2 = params1.copy()
    params2["image_dir"] = "/different/path"

    subset1 = DreamBoothSubset(**params1)
    subset2 = DreamBoothSubset(**params1)  # Same parameters
    subset3 = DreamBoothSubset(**params2)  # Different image_dir

    assert subset1 == subset2
    assert subset1 != subset3
    assert subset1 != "not a subset"  # Test with non-DreamBoothSubset object


def test_invalid_image_dir():
    """Test that initialization fails when image_dir is None."""
    params = {
        "image_dir": None,  # Invalid
        "is_reg": False,
        "class_tokens": "person",
        "caption_extension": "txt",
        "cache_info": True,
        "alpha_mask": False,
        "num_repeats": 1,
        "shuffle_caption": True,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": ",",
        "secondary_separator": ";",
        "enable_wildcard": False,
        "color_aug": False,
        "flip_aug": False,
        "face_crop_aug_range": None,
        "random_crop": False,
        "caption_dropout_rate": 0.0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0.0,
        "caption_prefix": "",
        "caption_suffix": "",
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    }

    with pytest.raises(AssertionError):
        DreamBoothSubset(**params)


class TestMultiBucketDataset:
    def test_multi_dataset_initialization(self):
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
            datasets=[dataset1, dataset2], batch_size=1, min_bucket_reso=256, max_bucket_reso=1024, bucket_reso_steps=64
        )

        # Check initialization
        assert len(multi_ds.datasets) == 2
        assert multi_ds.dataset_map == {}

    def test_sync_state_to_datasets(self):
        """Test that state is synchronized to all datasets."""
        # Create mock datasets
        dataset1 = MagicMock()
        dataset2 = MagicMock()

        # Create multi-bucket dataset
        multi_ds = MultiBucketDataset(
            datasets=[dataset1, dataset2],
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

    @pytest.mark.parametrize("file_count", [1, 3, 5])
    def test_multi_dataset_datasets(self, default_params, temp_image_text_files, file_count):
        # Create the temporary files
        files = temp_image_text_files(file_count)

        config = ConfigSanitizer(support_dreambooth=True, support_controlnet=True, support_finetuning=True, support_dropout=True)
        generator = BlueprintGenerator(config)
        blueprint = generator.generate({"datasets": [{"subsets": [{"image_dir": files["dir"]}]}]}, argparse.Namespace())
        train, val = generate_dataset_group_by_blueprint(blueprint.dataset_group)

        dataset = MultiBucketDataset(
            datasets=[train, train],
            batch_size=1,
            min_bucket_reso=256,
            max_bucket_reso=1024,
        )

        dataset.prepare_buckets()

        print(len(dataset))

        for item in dataset:
            print(item)

        assert dataset is None

        pass
