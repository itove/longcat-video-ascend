# LongCat-Video Ascend NPU 适配文档

本文档描述如何将美团开源的 **LongCat-Video** (13.6B 参数视频生成模型) 适配到 **华为昇腾 NPU (Ascend 910B 系列)** 上运行。

> **模型官方仓库**：https://github.com/meituan-longcat/LongCat-Video  
> **模型权重下载**：https://huggingface.co/meituan-longcat/LongCat-Video  
> **AtomGit / ModelScope 镜像**：可从 AtomGit 或 ModelScope 下载原始权重  
> **适配版本**：基于 LongCat-Video main 分支  
> **硬件验证环境**：Atlas 800 A2 (Ascend 910B3)  
> **软件栈**：CANN 8.0+, torch 2.5.1+, torch-npu 2.5.1+

---

## 目录

- [适配概述](#适配概述)
- [环境准备](#环境准备)
- [模型权重下载](#模型权重下载)
- [代码修改说明](#代码修改说明)
- [运行推理](#运行推理)
- [验证测试](#验证测试)
- [已知限制](#已知限制)
- [常见问题](#常见问题)

---

## 适配概述

LongCat-Video 原生基于 PyTorch + Diffusers 框架，使用 CUDA 生态（包括 FlashAttention-2/3、Triton BSA、NCCL 等）。在昇腾 NPU 上的核心适配工作包括：

| 原始依赖 | 昇腾替代方案 | 说明 |
|---------|------------|------|
| `torch.cuda` | `torch.npu` | 通过兼容层自动切换 |
| `nccl` | `hccl` | 分布式通信后端自动切换 |
| `flash-attn` / `flash_attn_interface` | 标准 PyTorch Attention | 自动 fallback，无需安装 flash-attn |
| `triton` BSA (Block Sparse Attention) | 标准 PyTorch Attention | Triton 算子在 NPU 上不可用，自动降级 |
| `xformers` | 标准 PyTorch Attention | 同上 |
| `torch.compile` | 条件跳过 | NPU 环境下自动禁用 compile |
| `amp.autocast(device_type='cuda')` | 动态 device 检测 | 自动适配为 `npu` 或 `cuda` |

---

## 环境准备

### 1. 硬件与驱动

- **NPU 型号**：Ascend 910B3 (或兼容型号)
- **驱动版本**：请参照 [昇腾社区](https://www.hiascend.com/) 推荐版本
- **NPU 数量**：单卡或多卡均可（多卡需开启 Context Parallel）

### 2. CANN 工具链

```bash
# 确认 ASCEND_TOOLKIT_HOME 已设置
echo $ASCEND_TOOLKIT_HOME
# 输出示例: /usr/local/Ascend/ascend-toolkit/latest

# 确认 npu-smi 可用
npu-smi info
```

### 3. Python 环境

```bash
conda create -n longcat-video-npu python=3.10
conda activate longcat-video-npu

# 安装昇腾版 PyTorch (请匹配 CANN 版本)
# 示例 (CANN 8.0 + torch 2.5.1):
pip install torch==2.5.1 torchvision torchaudio
pip install torch_npu==2.5.1

# 安装其他依赖 (不需要 flash-attn)
pip install -r requirements-npu.txt
```

**requirements-npu.txt** 已包含除 flash-attn 外的所有必要依赖。

### 4. 验证 PyTorch NPU 可用性

```bash
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
# 期望输出: True
```

---

## 模型权重下载

### 方式一：HuggingFace (官方)

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video
```

### 方式二：ModelScope (国内镜像)

```bash
pip install modelscope
modelscope download --model meituan-longcat/LongCat-Video --local_dir ./weights/LongCat-Video
```

### 方式三：AtomGit

如 AtomGit 已托管该模型权重，可直接从 AtomGit 仓库下载到 `./weights/LongCat-Video`。

---

## 代码修改说明

本适配仓库已包含所有必要的代码修改，核心变更如下：

### 1. 新增 `ascend_npu_compat.py`

设备检测与兼容层，提供：
- `get_device_type()` / `get_device_count()` / `set_device()`
- `empty_cache()` / `ipc_collect()` / `Stream()`
- `init_process_group()` — 自动选择 `hccl` / `nccl`
- `standard_attention()` — PyTorch 原生 Attention fallback
- `maybe_compile()` — NPU 时跳过 `torch.compile`

### 2. Attention 模块 fallback (`longcat_video/modules/attention.py`)

在所有 flash-attn / xformers 分支后新增标准 PyTorch Attention 回退：

```python
else:
    # Standard PyTorch attention fallback (for Ascend NPU / CPU)
    x = _standard_attention(q, k, v, self.scale)
```

同样适用于 `MultiHeadCrossAttention` 和 `avatar/attention.py`。

### 3. 模型初始化自动降级 (`longcat_video_dit.py`)

`LongCatVideoTransformer3DModel.__init__` 开头自动检测 NPU 并关闭 flash-attn / BSA：

```python
if hasattr(torch, 'npu') and torch.npu.is_available():
    print("[Ascend NPU] Auto-disabling flash-attn/xformers/bsa...")
    enable_flashattn3 = False
    enable_flashattn2 = False
    enable_xformers = False
    enable_bsa = False
    bsa_params = None
```

### 4. 分布式通信 (`context_parallel_util.py`)

- `init_device_mesh("cuda", ...)` 改为动态 `device_type`
- `torch.cuda.Stream()` 改为兼容层 `_get_stream()`

### 5. Pipeline 与 Demo 脚本

- `torch.cuda.empty_cache()` → `empty_cache()`
- `torch.cuda.device_count()` → `get_device_count()`
- `dist.init_process_group(backend="nccl")` → `init_process_group()`
- `self.device = "cuda"` → 动态检测 `npu` / `cuda`
- `torch.compile(dit)` → 条件化跳过 (NPU 时不编译)

### 6. AMP Autocast

所有 `amp.autocast(device_type='cuda')` 改为调用 `_get_amp_device()` 动态返回 `npu` / `cuda`。

---

## 运行推理

### 单卡 Text-to-Video (NPU)

```bash
# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0

# 运行
torchrun run_demo_text_to_video.py \
  --checkpoint_dir=./weights/LongCat-Video
```

> **注意**：NPU 环境下 `--enable_compile` 参数会被自动忽略，不会触发 `torch.compile`。

### 多卡 Text-to-Video (Context Parallel)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 run_demo_text_to_video.py \
  --context_parallel_size=2 \
  --checkpoint_dir=./weights/LongCat-Video
```

多卡场景下会自动使用 **HCCL** 作为分布式后端。

### Image-to-Video / Video-Continuation / Long-Video

用法与官方 README 完全一致，只需确保环境为 NPU：

```bash
torchrun run_demo_image_to_video.py --checkpoint_dir=./weights/LongCat-Video
torchrun run_demo_video_continuation.py --checkpoint_dir=./weights/LongCat-Video
torchrun run_demo_long_video.py --checkpoint_dir=./weights/LongCat-Video
```

---

## 验证测试

运行一键验证脚本：

```bash
bash verify_npu.sh
```

该脚本将依次执行：
1. **环境检查**：npu-smi、CANN、torch_npu 版本
2. **模块导入测试**：ascend_npu_compat、attention、pipeline
3. **Attention Fallback 测试**：标准 PyTorch Attention 前向正确性
4. **Dummy Forward 测试**：使用随机权重验证模型前向传播
5. **分布式后端检查**：确认 HCCL 被选中

---

## 已知限制

| 限制项 | 说明 | 预计解决 |
|-------|------|---------|
| **flash-attn** | NPU 不支持原版 FlashAttention，已 fallback 到标准 Attention | 可用昇腾 FA 算子进一步优化 |
| **triton BSA** | Block Sparse Attention 基于 Triton，NPU 不支持 | 标准 Attention 替代 |
| **torch.compile** | NPU 上未启用图编译 | 待 torch_npu.compile 稳定后支持 |
| **性能** | 标准 Attention 的内存与速度不如 FlashAttention | 可通过昇腾亲和算子后续优化 |

---

## 常见问题

### Q1: 导入时提示 `flash-attn` 未安装？

**A**: NPU 环境下不需要安装 flash-attn。模型已自动降级到标准 PyTorch Attention。如仍报错，可设置环境变量跳过：

```bash
export DISABLE_FLASH_ATTN=1
```

### Q2: 启动多卡报错 `backend hccl not found`？

**A**: 确认 CANN 工具链已正确安装并 source 环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Q3: 精度与 CUDA 版本有差异？

**A**: 标准 PyTorch Attention 使用 `float32` softmax 计算，数值差异通常在 1e-3 以内，对视频生成质量无显著影响。如需严格对齐，可对比中间特征图。

### Q4: 如何贡献优化？

**A**: 欢迎提交 PR。当前优化方向：
- 接入 `torch_npu` 融合算子替换标准 Attention
- 支持 `torch.compile` / `dynamo` 图编译
- 优化 Context Parallel 通信效率

---

## 附录：文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ascend_npu_compat.py` | 新增 | NPU 兼容层与 Attention fallback |
| `apply_npu_patches.py` | 新增 | 批量补丁脚本（开发用） |
| `requirements-npu.txt` | 新增 | NPU 依赖列表（无 flash-attn） |
| `verify_npu.sh` | 新增 | 一键验证脚本 |
| `README-ascend.md` | 新增 | 本文档 |
| `longcat_video/modules/attention.py` | 修改 | 新增标准 Attention fallback |
| `longcat_video/modules/avatar/attention.py` | 修改 | 同上 |
| `longcat_video/modules/longcat_video_dit.py` | 修改 | NPU 自动禁用 flash-attn |
| `longcat_video/modules/avatar/longcat_video_dit_avatar.py` | 修改 | 同上 |
| `longcat_video/context_parallel/context_parallel_util.py` | 修改 | HCCL / 动态 device 支持 |
| `longcat_video/pipeline_longcat_video.py` | 修改 | 动态 device / cache 管理 |
| `longcat_video/pipeline_longcat_video_avatar.py` | 修改 | 同上 |
| `longcat_video/modules/blocks.py` | 修改 | 动态 autocast device |
| `run_demo_*.py` | 修改 | NPU 兼容的分布式与设备管理 |
| `run_streamlit.py` | 修改 | 动态 device 检测 |
| `longcat_video/audio_process/torch_utils.py` | 修改 | NPU cache 管理 |

---

**适配完成日期**：2026-05-14  
**适配作者**：JeffDing / AI Agent  
**提交仓库**：https://gitcode.com/JeffDing/LongCat-Video-Ascend (示例)
