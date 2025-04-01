# Minimum Inference Code for FLUX

import argparse
import datetime
import math
import os
import random
from typing import Callable, List, Optional
import einops
import numpy as np

from tqdm import tqdm
from PIL import Image
import accelerate
from transformers import CLIPTextModel
from safetensors.torch import load_file

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
from networks import oft_flux

from library.flux_utils import bypass_flux_guidance, restore_flux_guidance

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, strategy_flux
from library.utils import setup_logging, str_to_dtype
from library.flux_train_utils import get_schedule

init_ipex()
import torch

setup_logging()
import logging

logger = logging.getLogger(__name__)

@torch.inference_mode()
def do_sample(
    accelerator: Optional[accelerate.Accelerator],
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    t5_attn_mask: Optional[torch.Tensor],
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
    neg_l_pooled: Optional[torch.Tensor] = None,
    neg_t5_out: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    logger.info(f"num_steps: {num_steps}")
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    # bypass guidance module
    if args.bypass_flux_guidance:
        bypass_flux_guidance(model)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast():
            x = flux_utils.denoise_with_cfg(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale=cfg_scale,
            )
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype):
            x = flux_utils.denoise_with_cfg(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale=cfg_scale,
            )

    return x


def generate_image(
    model,
    clip_l: CLIPTextModel,
    t5xxl,
    ae,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    negative_prompt: Optional[str],
    cfg_scale: float,
    iteration: int=None
):
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")

    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(image_width / 16)
    noise_dtype = torch.float32 if is_fp8(dtype) else dtype
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=device,
        dtype=noise_dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # prepare img and img ids

    # this is needed only for img2img
    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # if img.shape[0] == 1 and bs > 1:
    #     img = repeat(img, "1 ... -> bs ...", bs=bs)

    # txt2img only needs img_ids
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

    # prepare fp8 models
    if is_fp8(clip_l_dtype) and (not hasattr(clip_l, "fp8_prepared") or not clip_l.fp8_prepared):
        logger.info(f"prepare CLIP-L for fp8: set to {clip_l_dtype}, set embeddings to {torch.bfloat16}")
        clip_l.to(clip_l_dtype)  # fp8
        clip_l.text_model.embeddings.to(dtype=torch.bfloat16)
        clip_l.fp8_prepared = True

    if is_fp8(t5xxl_dtype) and (not hasattr(t5xxl, "fp8_prepared") or not t5xxl.fp8_prepared):
        logger.info(f"prepare T5xxl for fp8: set to {t5xxl_dtype}")

        def prepare_fp8(text_encoder, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    hidden_gelu = module.act(module.wi_0(hidden_states))
                    hidden_linear = module.wi_1(hidden_states)
                    hidden_states = hidden_gelu * hidden_linear
                    hidden_states = module.dropout(hidden_states)

                    hidden_states = module.wo(hidden_states)
                    return hidden_states

                return forward

            for module in text_encoder.modules():
                if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        t5xxl.to(t5xxl_dtype)
        prepare_fp8(t5xxl.encoder, torch.bfloat16)
        t5xxl.fp8_prepared = True

    # prepare embeddings
    logger.info("Encoding prompts...")
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)

    def encode(prpt: str):
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        with torch.no_grad():
            if is_fp8(clip_l_dtype):
                with accelerator.autocast():
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
            else:
                with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)

            if is_fp8(t5xxl_dtype):
                with accelerator.autocast():
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

    l_pooled, t5_out, txt_ids, t5_attn_mask = encode(prompt)
    if negative_prompt:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode(negative_prompt)
    else:
        neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None

    # NaN check
    if torch.isnan(l_pooled).any():
        raise ValueError("NaN in l_pooled")
    if torch.isnan(t5_out).any():
        raise ValueError("NaN in t5_out")

    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
        logger.info("Offloaded clip_l and t5xxl")
    device_utils.clean_memory()

    logger.info(f"Move Flux to {device}")
    model.move_to_device_except_swap_blocks(device)

    # generate image
    logger.info("Generating image...")
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    t5_attn_mask = t5_attn_mask.to(device) if args.apply_t5_attn_mask else None

    x = do_sample(
        accelerator,
        model,
        noise,
        img_ids,
        l_pooled,
        t5_out,
        txt_ids,
        steps,
        guidance,
        t5_attn_mask,
        is_schnell,
        device,
        flux_dtype,
        neg_l_pooled,
        neg_t5_out,
        neg_t5_attn_mask,
        cfg_scale,
    )
    if args.offload:
        model = model.cpu()
    # del model
    device_utils.clean_memory()

    # unpack
    x = x.float()
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

    # decode
    logger.info("Decoding image...")
    ae = ae.to(device)
    with torch.no_grad():
        if is_fp8(ae_dtype):
            with accelerator.autocast():
                x = ae.decode(x)
        else:
            with torch.autocast(device_type=device.type, dtype=ae_dtype):
                x = ae.decode(x)
    if args.offload:
        ae = ae.cpu()

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # restore guidance module
    if args.bypass_flux_guidance:
        restore_flux_guidance(model)
    
    # save image
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if iteration is None:
        output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    else:
        output_path = os.path.join(output_dir, f"{prompt}_{iteration:05d}.png")
    img.save(output_path)

    logger.info(f"Saved image to {output_path}")


if __name__ == "__main__":
    target_height = 768  # 1024
    target_width = 1360  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--ae", type=str, required=False)
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--prompt", type=str, help="prompt")
    parser.add_argument("--prompts_file", type=str, help="prompts file")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--clip_l_dtype", type=str, default=None, help="dtype for clip_l")
    parser.add_argument("--ae_dtype", type=str, default=None, help="dtype for ae")
    parser.add_argument("--t5xxl_dtype", type=str, default=None, help="dtype for t5xxl")
    parser.add_argument("--flux_dtype", type=str, default=None, help="dtype for flux")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument("--bypass_flux_guidance", action="store_true", help="Bypass flux guidance for flex")
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora_flux and lora_oft, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    def is_fp8(dt):
        return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

    dtype = str_to_dtype(args.dtype)
    clip_l_dtype = str_to_dtype(args.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(args.ae_dtype, dtype)
    flux_dtype = str_to_dtype(args.flux_dtype, dtype)

    logger.info(f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}")

    loading_device = "cpu" if args.offload else device

    use_fp8 = [is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None

    # load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    t5xxl.eval()

    # if is_fp8(clip_l_dtype):
    #     clip_l = accelerator.prepare(clip_l)
    # if is_fp8(t5xxl_dtype):
    #     t5xxl = accelerator.prepare(t5xxl)

    # DiT
    is_schnell, model = flux_utils.load_flow_model(args.ckpt_path, None, loading_device)
    model.eval()
    logger.info(f"Casting model to {flux_dtype}")
    model.to(dtype=flux_dtype)  # make sure model is dtype
    model.enable_block_swap(6, device)
    # if is_fp8(flux_dtype):
    #     model = accelerator.prepare(model)
    #     if args.offload:
    #         model = model.to("cpu")

    t5xxl_max_length = 256 if is_schnell else 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # AE
    ae = flux_utils.load_ae(args.ae, ae_dtype, loading_device)
    ae.eval()
    # if is_fp8(ae_dtype):
    #     ae = accelerator.prepare(ae)

    # LoRA
    lora_models: List[lora_flux.LoRANetwork] = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        is_lora = is_oft = False
        for key in weights_sd.keys():
            if key.startswith("lora"):
                is_lora = True
            if key.startswith("oft"):
                is_oft = True
            if is_lora or is_oft:
                break

        module = lora_flux if is_lora else oft_flux
        lora_model, _ = module.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd, True)

        if args.merge_lora_weights:
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        else:
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.eval()
            lora_model.to(device)

        lora_models.append(lora_model)

    if not args.interactive:
        if args.prompt is not None:
            prompts = args.prompt
        elif args.prompts_file is not None:
            with open(args.prompts_file) as f:
                prompts = f.readlines()

        progress = tqdm(desc="images", total=len(prompts))
        for i, prompt in enumerate(prompts):
            generate_image(
                model,
                clip_l,
                t5xxl,
                ae,
                prompt.strip(),
                args.seed,
                args.width,
                args.height,
                args.steps,
                args.guidance,
                args.negative_prompt,
                args.cfg_scale,
                i
            )
            progress.update(1)
    else:
        # loop for interactive
        width = target_width
        height = target_height
        steps = None
        guidance = args.guidance
        cfg_scale = args.cfg_scale

        while True:
            print(
                "Enter prompt (empty to exit). Options: --w <width> --h <height> --s <steps> --d <seed> --g <guidance> --m <multipliers for LoRA>"
                " --n <negative prompt>, `-` for empty negative prompt --c <cfg_scale>"
            )
            prompt = input()
            if prompt == "":
                break

            # parse options
            options = prompt.split("--")
            prompt = options[0].strip()
            seed = None
            negative_prompt = None
            for opt in options[1:]:
                try:
                    opt = opt.strip()
                    if opt.startswith("w"):
                        width = int(opt[1:].strip())
                    elif opt.startswith("h"):
                        height = int(opt[1:].strip())
                    elif opt.startswith("s"):
                        steps = int(opt[1:].strip())
                    elif opt.startswith("d"):
                        seed = int(opt[1:].strip())
                    elif opt.startswith("g"):
                        guidance = float(opt[1:].strip())
                    elif opt.startswith("m"):
                        mutipliers = opt[1:].strip().split(",")
                        if len(mutipliers) != len(lora_models):
                            logger.error(f"Invalid number of multipliers, expected {len(lora_models)}")
                            continue
                        for i, lora_model in enumerate(lora_models):
                            lora_model.set_multiplier(float(mutipliers[i]))
                    elif opt.startswith("n"):
                        negative_prompt = opt[1:].strip()
                        if negative_prompt == "-":
                            negative_prompt = ""
                    elif opt.startswith("c"):
                        cfg_scale = float(opt[1:].strip())
                except ValueError as e:
                    logger.error(f"Invalid option: {opt}, {e}")

            generate_image(model, clip_l, t5xxl, ae, prompt, seed, width, height, steps, guidance, negative_prompt, cfg_scale)

    logger.info("Done!")
