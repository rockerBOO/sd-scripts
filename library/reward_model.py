import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import CLIPModel, CLIPProcessor
from library import strategy_base


class CLIPRewardModel:
    def __init__(self, model_name_or_path="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)

        self.model.eval()
        self.accelerator: Accelerator | None = None  # Will be set after initialization

    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        return self

    def __call__(self, images: torch.Tensor, prompts: list[str]):
        """
        Args:
            images: [B, 3, H, W] tensor in [0, 1]
            prompts: List of prompts
        Returns:
            rewards: [B] similarity scores
        """
        # Move to compute device if currently on CPU
        original_device = next(self.model.parameters()).device
        should_offload = original_device.type == "cpu"

        if should_offload and self.accelerator is not None:
            self.model.to(self.accelerator.device)

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        if self.accelerator is not None:
            with self.accelerator.autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        # Cosine similarity as reward
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)

        similarity = (image_embeds * text_embeds).sum(dim=-1)

        # Offload back to CPU after computation
        if should_offload:
            self.model.to("cpu")
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

        return similarity


class CLIPRewardTokenHandler:
    """
    Handles token processing for CLIP reward models with support for long prompts.
    """

    def __init__(
        self,
        model,
        processor,
    ):
        """
        Initialize CLIP reward token handler.

        Args:
            model: CLIP model
            processor: CLIP processor for tokenization
        """
        self.model = model
        self.processor = processor

    @torch.no_grad()
    def compute_embeddings(self, prompts: list[str]) -> torch.Tensor:
        """
        Compute embeddings for prompts, handling length constraints via processor.

        Args:
            prompts: List of text prompts

        Returns:
            Tensor of text embeddings
        """

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()

        clip_tokens, _, _ = tokenize_strategy.tokenize(prompts)

        # Compute text embeddings
        text_embeddings = self.model.get_text_features(input_ids=clip_tokens)

        return text_embeddings

    def __call__(self, prompts: str | list[str]) -> torch.Tensor:
        """
        Convenience method for computing embeddings.

        Args:
            prompts: Single prompt or list of prompts

        Returns:
            Tensor of embeddings
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.compute_embeddings(prompts)
