import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from library.device_utils import clean_memory_on_device
from library.safetensors_utils import MemoryEfficientSafeOpen
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    Calculate the maximum representable value in FP8 format.
    Default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign). Only supports E4M3 and E5M2 with sign bit.

    Args:
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits (0 or 1)

    Returns:
        float: Maximum value representable in FP8 format
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"
    if exp_bits == 4 and mantissa_bits == 3 and sign_bits == 1:
        return torch.finfo(torch.float8_e4m3fn).max
    elif exp_bits == 5 and mantissa_bits == 2 and sign_bits == 1:
        return torch.finfo(torch.float8_e5m2).max
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits} with sign_bits={sign_bits}")


# The following is a manual calculation method (wrong implementation for E5M2), kept for reference.
"""
# Calculate exponent bias
bias = 2 ** (exp_bits - 1) - 1

# Calculate maximum mantissa value
mantissa_max = 1.0
for i in range(mantissa_bits - 1):
    mantissa_max += 2 ** -(i + 1)

# Calculate maximum value
max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

return max_value
"""


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value, can_modify_inplace=False):
    """
    Quantize a tensor to FP8 format using PyTorch's native FP8 dtype support.
    Memory-efficient version using in-place operations where possible.

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        fp8_dtype (torch.dtype): Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
        max_value (float): Maximum representable value in FP8
        min_value (float): Minimum representable value in FP8
        can_modify_inplace (bool): If True, modifies input tensor in-place to save memory

    Returns:
        torch.Tensor: Quantized tensor in FP8 format
    """
    # If we can't modify in-place, clone first
    if not can_modify_inplace:
        tensor = tensor.clone()

    # Convert to float32 for division (in-place if already a copy)
    tensor = tensor.to(torch.float32)

    # Use in-place operations
    tensor.div_(scale)
    tensor.nan_to_num_(0.0)  # handle NaN values in-place

    # Clamp tensor to range (in-place)
    tensor.clamp_(min=min_value, max=max_value)

    # Convert to FP8 dtype (creates new tensor)
    tensor = tensor.to(fp8_dtype)

    return tensor


def optimize_state_dict_with_fp8(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[list[str]] = None,
    exclude_layer_keys: Optional[list[str]] = None,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    move_to_device: bool = False,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
    fp8_quantize_batch_size: Optional[int] = None,
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format. The state dict is modified in-place.
    This function is a static version of load_safetensors_with_fp8_optimization without loading from files.

    Args:
        state_dict (dict): State dict to optimize, replaced in-place
        calc_device (str): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Create optimized state dict
    optimized_count = 0

    # Enumerate target keys
    target_state_dict_keys = []
    for key in state_dict.keys():
        # Check if it's a weight key and matches target patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    # Create batches for target keys
    if fp8_quantize_batch_size is None or fp8_quantize_batch_size == 0:
        # Process all keys at once (original behavior)
        batches = [target_state_dict_keys]
    else:
        # Split into batches
        batches = [
            target_state_dict_keys[i:i + fp8_quantize_batch_size]
            for i in range(0, len(target_state_dict_keys), fp8_quantize_batch_size)
        ]

    # Process each batch
    for batch_idx, batch_keys in enumerate(batches):
        batch_desc = f"Quantizing batch {batch_idx+1}/{len(batches)}"
        for key in tqdm(batch_keys, desc=batch_desc, leave=False):
            value = state_dict[key]

            # Save original device and dtype
            original_device = value.device
            original_dtype = value.dtype

            # Move to calculation device and track if we created a new tensor
            moved_to_different_device = False
            if calc_device is not None and str(calc_device) != str(original_device):
                value = value.to(calc_device)
                moved_to_different_device = True

            # can_modify_inplace=True only if we moved to different device (created new tensor)
            quantized_weight, scale_tensor = quantize_weight(
                key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size, can_modify_inplace=moved_to_different_device
            )

            # Add to state dict using original key for weight and new key for scale
            fp8_key = key  # Maintain original key
            scale_key = key.replace(".weight", ".scale_weight")

            if not move_to_device:
                quantized_weight = quantized_weight.to(original_device)

            # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
            scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

            state_dict[fp8_key] = quantized_weight
            state_dict[scale_key] = scale_tensor

            # Explicitly delete moved tensor to free memory immediately
            # Note: value may point to same object as state_dict[key] before assignment,
            # but after assignment to fp8_key, we can safely delete the reference
            del value

            optimized_count += 1

        # Clean up GPU memory after each batch
        if calc_device is not None and len(batch_keys) > 0:
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def quantize_weight(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    max_value: float,
    min_value: float,
    quantization_mode: str = "block",
    block_size: int = 64,
    can_modify_inplace: bool = False,
):
    original_shape = tensor.shape

    # Determine quantization mode
    if quantization_mode == "block":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor
        else:
            out_features, in_features = tensor.shape
            if in_features % block_size != 0:
                quantization_mode = "channel"  # fallback to per-channel
                logger.warning(
                    f"Layer {key} with shape {tensor.shape} is not divisible by block_size {block_size}, fallback to per-channel quantization."
                )
            else:
                num_blocks = in_features // block_size
                tensor = tensor.contiguous().view(out_features, num_blocks, block_size)  # [out, num_blocks, block_size]
    elif quantization_mode == "channel":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor

    # Calculate scale factor (per-tensor or per-output-channel with percentile or max)
    # value shape is expected to be [out_features, in_features] for Linear weights
    if quantization_mode == "channel" or quantization_mode == "block":
        # row-wise percentile to avoid being dominated by outliers
        # result shape: [out_features, 1] or [out_features, num_blocks, 1]
        scale_dim = 1 if quantization_mode == "channel" else 2
        abs_w = torch.abs(tensor)

        # shape: [out_features, 1] or [out_features, num_blocks, 1]
        row_max = torch.max(abs_w, dim=scale_dim, keepdim=True).values
        scale = row_max / max_value

    else:
        # per-tensor
        tensor_max = torch.max(torch.abs(tensor).view(-1))
        scale = tensor_max / max_value

    # numerical safety
    scale = torch.clamp(scale, min=1e-8)
    scale = scale.to(torch.float32)  # ensure scale is in float32 for division

    # Quantize weight to FP8 (scale can be scalar or [out,1], broadcasting works)
    # can_modify_inplace=True saves memory by not cloning the tensor
    quantized_weight = quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value, can_modify_inplace=can_modify_inplace)

    # If block-wise, restore original shape
    if quantization_mode == "block":
        quantized_weight = quantized_weight.view(original_shape)  # restore to original shape [out, in]

    return quantized_weight, scale


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
    fp8_quantize_batch_size: Optional[int] = None,
) -> dict:
    """
    Load weight tensors from safetensors files and merge LoRA weights into the state dict with explicit FP8 optimization.

    Args:
        model_files (list[str]): List of model files to load
        calc_device (str or torch.device): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target for optimization (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude from optimization
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device
        weight_hook (callable, optional): Function to apply to each weight tensor before optimization
        quantization_mode (str): Quantization mode, "tensor", "channel", or "block"
        block_size (int, optional): Block size for block-wise quantization (used if quantization_mode is "block")

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Define function to determine if a key is a target key. target means fp8 optimization, not for weight hook.
    def is_target_key(key):
        # Check if weight key matches target patterns and does not match exclude patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        return is_target and not is_excluded

    # Create optimized state dict
    optimized_count = 0
    state_dict = {}

    # Process each file
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = f.keys()

            # Collect target keys that need FP8 quantization for this file
            target_keys_in_file = []
            non_target_keys = []

            for key in keys:
                if is_target_key(key):
                    target_keys_in_file.append(key)
                else:
                    non_target_keys.append(key)

            # Process non-target keys first (these don't need batching)
            for key in tqdm(non_target_keys, desc=f"Loading non-FP8 keys from {os.path.basename(model_file)}", unit="key", leave=False):
                value = f.get_tensor(key)
                original_device = value.device

                if weight_hook is not None:
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                value = value.to(target_device)
                state_dict[key] = value

<<<<<<< Updated upstream
                # Move to calculation device
                if calc_device is not None:
                    value = value.to(calc_device)

                original_dtype = value.dtype
                quantized_weight, scale_tensor = quantize_weight(
                    key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
                )
=======
            # Create batches for target keys
            if fp8_quantize_batch_size is None or fp8_quantize_batch_size == 0:
                # Process all target keys at once (original behavior)
                batches = [target_keys_in_file]
            else:
                # Split into batches
                batches = [
                    target_keys_in_file[i:i + fp8_quantize_batch_size]
                    for i in range(0, len(target_keys_in_file), fp8_quantize_batch_size)
                ]

            # Process each batch
            for batch_idx, batch_keys in enumerate(batches):
                batch_desc = f"Quantizing batch {batch_idx+1}/{len(batches)} from {os.path.basename(model_file)}"
                for key in tqdm(batch_keys, desc=batch_desc, unit="key", leave=False):
                    value = f.get_tensor(key)

                    # Save original device
                    original_device = value.device  # usually cpu

                    if weight_hook is not None:
                        # Apply weight hook if provided
                        value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                    original_dtype = value.dtype

                    if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
                        logger.warning(
                            f"Skipping FP8 quantization for key {key} as it is already in FP8 format ({original_dtype}). "
                            "Loading checkpoint as-is without re-quantization."
                        )
                        target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                        value = value.to(target_device)
                        state_dict[key] = value
                        continue

                    # Move to calculation device and track if we created a new tensor
                    # Check if we have enough GPU memory first
                    moved_to_different_device = False
                    quantize_device = value.device  # Default to current device (usually CPU)

                    if calc_device is not None and str(calc_device) != str(value.device):
                        # Check if target device is CUDA and if we have enough memory
                        if "cuda" in str(calc_device):
                            required_memory = value.numel() * value.element_size() * 2  # Estimate: weight + float32 copy
                            available_memory = torch.cuda.mem_get_info(calc_device)[0]  # Free memory in bytes

                            if available_memory < required_memory * 1.2:  # 20% safety margin
                                logger.warning(
                                    f"Insufficient GPU memory for layer {key} ({required_memory/1024**2:.1f} MB needed, "
                                    f"{available_memory/1024**2:.1f} MB available). Quantizing on CPU instead."
                                )
                                quantize_device = value.device  # Keep on CPU
                            else:
                                value = value.to(calc_device)
                                quantize_device = calc_device
                                moved_to_different_device = True
                        else:
                            value = value.to(calc_device)
                            quantize_device = calc_device
                            moved_to_different_device = True

                    # can_modify_inplace=True only if we moved to different device (created new tensor)
                    quantized_weight, scale_tensor = quantize_weight(
                        key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size, can_modify_inplace=moved_to_different_device
                    )

                    # Explicitly delete original tensor to free GPU memory immediately
                    del value

                    # Add to state dict using original key for weight and new key for scale
                    fp8_key = key  # Maintain original key
                    scale_key = key.replace(".weight", ".scale_weight")
                    assert fp8_key != scale_key, "FP8 key and scale key must be different"
>>>>>>> Stashed changes

                    if not move_to_device:
                        # Move FP8 result back to CPU to free GPU memory for next layer
                        quantized_weight = quantized_weight.to(original_device)
                        scale_tensor_device = original_device
                    else:
                        scale_tensor_device = quantized_weight.device

                    # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
                    scale_tensor = scale_tensor.to(dtype=original_dtype, device=scale_tensor_device)

                    state_dict[fp8_key] = quantized_weight
                    state_dict[scale_key] = scale_tensor

                    # Clean up GPU memory after each layer if quantizing on GPU but moving results to CPU
                    # This prevents memory buildup when processing layers incrementally
                    if calc_device is not None and not move_to_device:
                        clean_memory_on_device(calc_device)

                    optimized_count += 1

                # Clean up GPU memory after batch
                if calc_device is not None and len(batch_keys) > 0:
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """
    Patched forward method for Linear layers with FP8 weights.

    Args:
        self: Linear layer instance
        x (torch.Tensor): Input tensor
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)
        max_value (float): Maximum value for FP8 quantization. If None, no quantization is applied for input tensor.

    Returns:
        torch.Tensor: Result of linear transformation
    """
    if use_scaled_mm:
        # **not tested**
        # _scaled_mm only works for per-tensor scale for now (per-channel scale does not work in certain cases)
        if self.scale_weight.ndim != 1:
            raise ValueError("scaled_mm only supports per-tensor scale_weight for now.")

        input_dtype = x.dtype
        original_weight_dtype = self.scale_weight.dtype
        target_dtype = self.weight.dtype
        # assert x.ndim == 3, "Input tensor must be 3D (batch_size, seq_len, hidden_dim)"

        if max_value is None:
            # no input quantization
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # calculate scale factor for input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # quantize input tensor to FP8: this seems to consume a lot of memory
            fp8_max_value = torch.finfo(target_dtype).max
            fp8_min_value = torch.finfo(target_dtype).min
            x = quantize_fp8(x, scale_x, target_dtype, fp8_max_value, fp8_min_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            # float32 is not supported with bias in scaled_mm
            o = torch._scaled_mm(x, weight, out_dtype=original_weight_dtype, bias=self.bias, scale_a=scale_x, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        o = o.reshape(original_shape[0], original_shape[1], -1) if x.ndim == 3 else o.reshape(original_shape[0], -1)
        return o.to(input_dtype)

    else:
        # Dequantize the weight
        original_dtype = self.scale_weight.dtype
        if self.scale_weight.ndim < 3:
            # per-tensor or per-channel quantization, we can broadcast
            dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
        else:
            # block-wise quantization, need to reshape weight to match scale shape for broadcasting
            out_features, num_blocks, _ = self.scale_weight.shape
            dequantized_weight = self.weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
            dequantized_weight = dequantized_weight * self.scale_weight
            dequantized_weight = dequantized_weight.view(self.weight.shape)

        # Perform linear transformation
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias)
        else:
            output = F.linear(x, dequantized_weight)

        return output


def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    # # Calculate FP8 float8_e5m2 max value
    # max_value = calculate_fp8_maxval(5, 2)
    max_value = None  # do not quantize input tensor

    # Find all scale keys to identify FP8-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # Enumerate patched layers
    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

        # Store scale shape information
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_count = 0

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # register the scale_weight as a buffer to load the state_dict
            # module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))
            scale_shape = scale_shape_info[name]
            module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=module.weight.dtype))

            # Create a new forward method with the patched version.
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model
