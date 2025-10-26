import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.src.open_clip.transform import InterpolationMode
from transformers import CLIPModel, CLIPProcessor

from library import strategy_base


class CLIPRewardModel(nn.Module):
    def __init__(self, model_name_or_path, accelerator: Accelerator):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)

        self.accelerator = accelerator
        self.model = self.model.eval()

    def forward(self, images: torch.Tensor, prompts: list[str]):
        """
        Args:
            images: [B, 3, H, W] tensor in [0, 1]
            prompts: List of prompts
        Returns:
            rewards: [B] similarity scores
        """
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False,
        ).to(self.accelerator.device)

        if self.accelerator is not None:
            with self.accelerator.autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        # Cosine similarity as reward
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)

        similarity = (image_embeds * text_embeds).sum(dim=-1)

        return similarity

    #### implment the SRP in CFG-like function；we find the （1-k)*neg + k *pos is less stable, we change it to (1+k)*pos-neg
    def SRP_cfg(self, prompts: str | list[str], neg_prompts: str | list[str], images: torch.Tensor, k: torch.Tensor):
        # Extract image features and text features for positive and negative prompts

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        positive_text_inputs, _, _ = tokenize_strategy.tokenize(prompts)
        negative_text_inputs, _, _ = tokenize_strategy.tokenize(neg_prompts)

        size = 224

        # 1. Resize shorter side to 224, preserving aspect ratio
        images = TF.resize(images, size=size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        # 2. Center crop to 224x224
        images = TF.center_crop(images, output_size=[size, size])

        # 3. Normalize with CLIP's mean and std
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        images_normalized = TF.normalize(images, mean=mean, std=std)

        # 3. Create the input dict that CLIP expects
        image_inputs = {"pixel_values": images_normalized}

        # Image embedding
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        # Extract image features and text features for positive and negative prompts
        text_embs = self.model.get_text_features(**positive_text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs_neg = self.model.get_text_features(**negative_text_inputs)
        text_embs_neg = text_embs_neg / text_embs_neg.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()

        # Compute raw cosine similarities for metrics
        pos_similarity = (text_embs @ image_embs.T).diag()
        neg_similarity = (text_embs_neg @ image_embs.T).diag()

        # Compute the reward based on the similarity
        scores = logit_scale * ((k + 1) * text_embs - text_embs_neg) @ image_embs.T
        scores = scores.diag()

        return scores, pos_similarity, neg_similarity


# class HPSRewardModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, images: torch.Tensor, prompts: list[str]):
#         """
#         Args:
#             images: [B, 3, H, W] tensor in [0, 1]
#             prompts: List of prompts
#         Returns:
#             rewards: [B] similarity scores
#         """
#
#         # Convert each image in the batch to PIL
#         to_pil = transforms.ToPILImage()
#         pil_images = [to_pil(img) for img in images]
#
#         scores = hpsv2.score(pil_images, prompt="", hps_version="v2.1")
#
#         return scores


class HPSRewardModel(torch.nn.Module):
    def __init__(self, model_path: str, accelerator: Accelerator):
        super().__init__()
        self.model_path = model_path
        self.accelerator = accelerator
        hpsv2_model, hpsv2_token, hpsv2_pre = self.build_reward_model()
        self.model = hpsv2_model
        self.token = hpsv2_token

        #### differentiable preprocessor
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224

        def _transform():
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(crop_size),
                    transforms.Normalize(std=image_std, mean=image_mean),
                ]
            )
            return transform

        self.vis_pre = _transform()

    def build_reward_model(self):
        model, preprocess_train, reprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint["state_dict"])
        text_processor = get_tokenizer("ViT-H-14")

        return model, text_processor, preprocess_train

    #### implment the SRP in CFG-like function；we find the （1-k)*neg + k *pos is less stable. therefore, we change it to (1+k)*pos-neg
    def SRP_cfg(self, prompt, neg_prompt, images, k):
        image = self.vis_pre(images).to(self.accelerator.device, non_blocking=True)
        text = self.token(prompt).to(self.accelerator.device, non_blocking=True)
        neg_text = self.token(neg_prompt).to(self.accelerator.device, non_blocking=True)
        with self.accelerator.autocast():
            # Extract image features and text features for positive and negative prompts
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            text_features_neg = self.model.encode_text(neg_text, normalize=True)

            # Compute the reward based on the similarity
            logits_per_image = image_features @ ((1 + k) * text_features.T - text_features_neg.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score

    def SRP(self, prompt, images, k):
        image = self.vis_pre(images).to(self.accelerator.device, non_blocking=True)
        text = self.token(prompt).to(self.accelerator.device, non_blocking=True)
        with self.accelerator.autocast():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            logits_per_image = image_features @ (k * text_features.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score
