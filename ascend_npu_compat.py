"""
Ascend NPU Compatibility Layer for LongCat-Video
Provides device-agnostic utilities to run LongCat-Video on Huawei Ascend NPUs.
"""

import os
import sys
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device_type():
    """Return 'npu', 'cuda', or 'cpu' based on availability."""
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return "npu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_count():
    """Return number of accelerators (NPU or CUDA)."""
    device_type = get_device_type()
    if device_type == "npu":
        import torch_npu
        return torch_npu.npu.device_count()
    elif device_type == "cuda":
        return torch.cuda.device_count()
    return 1


def set_device(local_rank):
    """Set current accelerator device."""
    device_type = get_device_type()
    if device_type == "npu":
        import torch_npu
        torch_npu.npu.set_device(local_rank)
    elif device_type == "cuda":
        torch.cuda.set_device(local_rank)


def empty_cache():
    """Clear accelerator cache."""
    device_type = get_device_type()
    if device_type == "npu":
        import torch_npu
        torch_npu.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def ipc_collect():
    """Collect IPC memory (CUDA only; no-op on NPU)."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.ipc_collect()


def get_current_device():
    """Return current device index."""
    device_type = get_device_type()
    if device_type == "npu":
        import torch_npu
        return torch_npu.npu.current_device()
    elif device_type == "cuda":
        return torch.cuda.current_device()
    return 0


def current_stream():
    """Return current accelerator stream."""
    device_type = get_device_type()
    if device_type == "npu":
        import torch_npu
        return torch_npu.npu.current_stream()
    elif device_type == "cuda":
        return torch.cuda.current_stream()
    return None


def Stream():
    """Return a new accelerator stream."""
    device_type = get_device_type()
    if device_type == "npu":
        import torch_npu
        return torch_npu.npu.Stream()
    elif device_type == "cuda":
        return torch.cuda.Stream()
    return None


def get_device_capability():
    """Return device capability tuple (NPU returns dummy high value)."""
    device_type = get_device_type()
    if device_type == "npu":
        return (9, 0)
    elif device_type == "cuda":
        return torch.cuda.get_device_capability()
    return (0, 0)


def is_available():
    """Check if any accelerator is available."""
    return get_device_type() in ("npu", "cuda")


# ---------------------------------------------------------------------------
# Distributed Backend
# ---------------------------------------------------------------------------

def get_distributed_backend():
    """Return 'hccl' for NPU, 'nccl' for CUDA, 'gloo' for CPU."""
    device_type = get_device_type()
    if device_type == "npu":
        return "hccl"
    elif device_type == "cuda":
        return "nccl"
    return "gloo"


def init_process_group(backend=None, **kwargs):
    """Initialize distributed process group with correct backend."""
    if backend is None:
        backend = get_distributed_backend()
    return dist.init_process_group(backend=backend, **kwargs)


# ---------------------------------------------------------------------------
# Autocast Helpers
# ---------------------------------------------------------------------------

def autocast(device_type=None, dtype=None, enabled=True):
    """Context manager for AMP autocast with dynamic device type."""
    if device_type is None:
        device_type = get_device_type()
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


# ---------------------------------------------------------------------------
# torch.compile Helper
# ---------------------------------------------------------------------------

def maybe_compile(model, enable=True, **kwargs):
    """Conditionally compile model; disabled on NPU by default."""
    if not enable:
        return model
    device_type = get_device_type()
    if device_type == "npu":
        # torch.compile on NPU may not be fully supported; skip for now
        return model
    return torch.compile(model, **kwargs)


# ---------------------------------------------------------------------------
# Attention Fallback (replaces flash-attn / triton BSA on NPU)
# ---------------------------------------------------------------------------

def standard_attention(q, k, v, scale=None, is_causal=False):
    """
    Pure PyTorch scaled dot-product attention fallback.
    Input shapes: [B, H, S, D]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    return out


def standard_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=None):
    """
    Variable-length attention fallback using standard PyTorch attention
    by processing each sequence independently.
    Inputs:
      q,k,v: [total_tokens, H, D] (flattened across batch)
      cu_seqlens_q/k: cumulative sequence lengths
    Returns: [total_tokens, H, D]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    batch_size = cu_seqlens_q.shape[0] - 1
    H, D = q.shape[1], q.shape[2]
    outputs = []
    for i in range(batch_size):
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()
        qi = q[start_q:end_q].unsqueeze(0).transpose(1, 2)  # [1, H, Sq, D]
        ki = k[start_k:end_k].unsqueeze(0).transpose(1, 2)  # [1, H, Sk, D]
        vi = v[start_k:end_k].unsqueeze(0).transpose(1, 2)  # [1, H, Sk, D]
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(qi.dtype)
        oi = torch.matmul(attn, vi).transpose(1, 2).squeeze(0)  # [Sq, H, D]
        outputs.append(oi)
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Monkey-patch helpers for dynamic device switching in pipelines
# ---------------------------------------------------------------------------

def patch_pipeline_device(pipeline_obj, device_attr_name="device"):
    """Patch a pipeline object to use the correct accelerator device."""
    device_type = get_device_type()
    if hasattr(pipeline_obj, device_attr_name):
        setattr(pipeline_obj, device_attr_name, device_type)


# ---------------------------------------------------------------------------
# Model Config Patcher
# ---------------------------------------------------------------------------

def patch_model_config_for_npu(config_path: str):
    """
    Optionally patch model config.json to disable flash-attn on NPU.
    This is a no-op if config doesn't exist or we're on CUDA.
    """
    import json
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    changed = False
    if get_device_type() == "npu":
        for key in ["enable_flashattn2", "enable_flashattn3", "enable_bsa", "enable_xformers"]:
            if config.get(key, False):
                config[key] = False
                changed = True
    if changed:
        backup_path = config_path + ".backup"
        if not os.path.exists(backup_path):
            os.rename(config_path, backup_path)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def restore_model_config(config_path: str):
    """Restore original model config from backup."""
    backup_path = config_path + ".backup"
    if os.path.exists(backup_path):
        os.replace(backup_path, config_path)
