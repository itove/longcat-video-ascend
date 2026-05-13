#!/bin/bash
# Ascend NPU Adaptation Verification Script for LongCat-Video
set -e

echo "=========================================="
echo "LongCat-Video Ascend NPU Verification"
echo "=========================================="

# -------------------------------
# 1. Environment Check
# -------------------------------
echo ""
echo "=== 1. Environment Check ==="

# NPU driver check
if command -v npu-smi &> /dev/null; then
    echo "[PASS] npu-smi found"
    npu-smi info || true
else
    echo "[WARN] npu-smi not found in PATH"
fi

# CANN check
if [ -n "$ASCEND_TOOLKIT_HOME" ]; then
    echo "[PASS] ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"
else
    echo "[WARN] ASCEND_TOOLKIT_HOME not set"
fi

# Python package check
echo "Checking Python packages..."
python3 -c "import torch; print('[INFO] torch version:', torch.__version__)" || echo "[FAIL] torch not installed"
python3 -c "import torch_npu; print('[INFO] torch_npu version:', torch_npu.__version__); print('[INFO] NPU available:', torch_npu.npu.is_available())" || echo "[FAIL] torch_npu not installed"
python3 -c "import transformers; print('[INFO] transformers version:', transformers.__version__)" || echo "[FAIL] transformers not installed"
python3 -c "import diffusers; print('[INFO] diffusers version:', diffusers.__version__)" || echo "[FAIL] diffusers not installed"
python3 -c "import einops; print('[INFO] einops available')" || echo "[FAIL] einops not installed"

# -------------------------------
# 2. Module Import Test
# -------------------------------
echo ""
echo "=== 2. Module Import Test ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from ascend_npu_compat import get_device_type, get_device_count, empty_cache, set_device
print('[PASS] ascend_npu_compat imported')
print('[INFO] Device type:', get_device_type())
print('[INFO] Device count:', get_device_count())
" || echo "[FAIL] ascend_npu_compat import failed"

python3 -c "
import sys
sys.path.insert(0, '.')
from longcat_video.context_parallel.context_parallel_util import init_context_parallel, get_cp_size
print('[PASS] context_parallel_util imported')
" || echo "[FAIL] context_parallel_util import failed"

python3 -c "
import sys
sys.path.insert(0, '.')
from longcat_video.modules.attention import Attention, MultiHeadCrossAttention
print('[PASS] attention module imported')
" || echo "[FAIL] attention import failed"

python3 -c "
import sys
sys.path.insert(0, '.')
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
print('[PASS] LongCatVideoTransformer3DModel imported')
" || echo "[FAIL] LongCatVideoTransformer3DModel import failed"

python3 -c "
import sys
sys.path.insert(0, '.')
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
print('[PASS] LongCatVideoPipeline imported')
" || echo "[FAIL] LongCatVideoPipeline import failed"

# -------------------------------
# 3. Attention Fallback Test
# -------------------------------
echo ""
echo "=== 3. Attention Fallback Test ==="
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from ascend_npu_compat import standard_attention

B, H, S, D = 2, 8, 64, 64
q = torch.randn(B, H, S, D)
k = torch.randn(B, H, S, D)
v = torch.randn(B, H, S, D)
out = standard_attention(q, k, v)
assert out.shape == (B, H, S, D)
print('[PASS] standard_attention output shape:', out.shape)
" || echo "[FAIL] standard_attention test failed"

# -------------------------------
# 4. Dummy Model Forward Test (no weights)
# -------------------------------
echo ""
echo "=== 4. Dummy Model Forward Test ==="
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

# Determine device
device = 'npu' if (hasattr(torch, 'npu') and torch.npu.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu')
print('[INFO] Using device:', device)

model = LongCatVideoTransformer3DModel(
    in_channels=16,
    out_channels=16,
    hidden_size=256,
    depth=2,
    num_heads=8,
    caption_channels=256,
    enable_flashattn2=False,
    enable_flashattn3=False,
    enable_xformers=False,
    enable_bsa=False,
).to(device)

B, C, T, H, W = 1, 16, 1, 16, 16
x = torch.randn(B, C, T, H, W, device=device)
t = torch.tensor([0.5], device=device)
y = torch.randn(B, 32, 256, device=device)

with torch.no_grad():
    out = model(x, t, y)
assert out.shape == x.shape
print('[PASS] Dummy forward pass succeeded, output shape:', out.shape)
" || echo "[FAIL] Dummy forward test failed"

# -------------------------------
# 5. Distributed Backend Check
# -------------------------------
echo ""
echo "=== 5. Distributed Backend Check ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from ascend_npu_compat import get_distributed_backend
backend = get_distributed_backend()
print('[INFO] Distributed backend:', backend)
if backend == 'hccl':
    print('[PASS] HCCL backend selected for NPU')
elif backend == 'nccl':
    print('[PASS] NCCL backend selected for CUDA')
else:
    print('[INFO] Gloo backend selected for CPU')
"

# -------------------------------
# Summary
# -------------------------------
echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
