import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # Handle shape mismatches by broadcasting or slicing appropriately
    if x.shape != shift.shape or x.shape != scale.shape:
        # Get the minimum sequence length to avoid index errors
        min_seq_len = min(x.shape[1], shift.shape[1], scale.shape[1])
        
        # Slice all tensors to the same sequence length
        x = x[:, :min_seq_len]
        shift = shift[:, :min_seq_len] 
        scale = scale[:, :min_seq_len]
    
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # Handle shape mismatches before calling modulate
    print("input into module_fused - shapes:", x.shape, shift.shape, scale.shape)
    if x.shape[1] != shift.shape[1] or x.shape[1] != scale.shape[1]:
        # Get the minimum sequence length
        min_seq_len = min(x.shape[1], shift.shape[1], scale.shape[1])
        print("Slicing to min_seq_len:", min_seq_len)
        # Slice all tensors to the same sequence length
        x = x[:, :min_seq_len]
        print("x shape after slicing:", x.shape)
        shift = shift[:, :min_seq_len]
        scale = scale[:, :min_seq_len]
    print("Inside modulate_fused - shapes:", x.shape, shift.shape, scale.shape)
    return x * (1 + scale) + shift