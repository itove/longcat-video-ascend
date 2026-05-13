#!/usr/bin/env python3
"""
Batch patch LongCat-Video codebase for Ascend NPU compatibility.
Run from repo root: python apply_npu_patches.py
"""

import os
import re


def patch_file(filepath, replacements):
    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} not found")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content == original:
        print(f"[SKIP] No changes in {filepath}")
        return
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[DONE] Patched {filepath}")


def patch_demo_script(filepath):
    replacements = [
        # torch_gc
        (
            "def torch_gc():\n    torch.cuda.empty_cache()\n    torch.cuda.ipc_collect()",
            "import sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nfrom ascend_npu_compat import empty_cache, ipc_collect, get_device_count, set_device, init_process_group\n\ndef torch_gc():\n    empty_cache()\n    ipc_collect()",
        ),
        # device setup
        (
            "num_gpus = torch.cuda.device_count()",
            "num_gpus = get_device_count()",
        ),
        (
            "torch.cuda.set_device(local_rank)",
            "set_device(local_rank)",
        ),
        # distributed backend
        (
            'dist.init_process_group(backend="nccl"',
            'init_process_group(',
        ),
    ]
    patch_file(filepath, replacements)


def patch_pipeline(filepath):
    replacements = [
        (
            "def torch_gc():\n    torch.cuda.empty_cache()\n    torch.cuda.ipc_collect()",
            "import sys\nimport os\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))\nfrom ascend_npu_compat import empty_cache, ipc_collect\n\ndef torch_gc():\n    empty_cache()\n    ipc_collect()",
        ),
        (
            'self.device = "cuda"',
            'self.device = "npu" if empty_cache.__module__.startswith("ascend") else ("cuda" if torch.cuda.is_available() else "cpu")',
        ),
    ]
    # Use regex for device line if simple replace fails
    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} not found")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content

    # torch_gc patch
    content = content.replace(
        "def torch_gc():\n    torch.cuda.empty_cache()\n    torch.cuda.ipc_collect()",
        "import sys\nimport os\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))\nfrom ascend_npu_compat import empty_cache, ipc_collect\n\ndef torch_gc():\n    empty_cache()\n    ipc_collect()",
    )

    # device patch - detect NPU availability
    device_re = re.compile(r'self\.device\s*=\s*"cuda"')
    if device_re.search(content):
        content = device_re.sub(
            'self.device = "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")',
            content,
        )

    # torch.cuda.empty_cache() standalone
    content = content.replace("torch.cuda.empty_cache()", "empty_cache()")

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[DONE] Patched {filepath}")
    else:
        print(f"[SKIP] No changes in {filepath}")


def patch_autocast(filepath):
    """Replace amp.autocast(device_type='cuda') with dynamic device detection."""
    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} not found")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content

    # Replace explicit cuda autocast
    content = re.sub(
        r"with amp\.autocast\(device_type='cuda'",
        "with amp.autocast(device_type=_get_amp_device()",
        content,
    )
    content = re.sub(
        r"with amp\.autocast\('cuda'",
        "with amp.autocast(_get_amp_device()",
        content,
    )

    if content != original:
        # Add helper import if missing
        helper = """
import torch
"""
        # Insert helper function after imports
        lines = content.splitlines()
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_idx = i + 1
        helper_func = "\ndef _get_amp_device():\n    if hasattr(torch, 'npu') and torch.npu.is_available():\n        return 'npu'\n    if torch.cuda.is_available():\n        return 'cuda'\n    return 'cpu'\n"
        lines.insert(import_idx, helper_func)
        content = "\n".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[DONE] Patched {filepath}")
    else:
        print(f"[SKIP] No changes in {filepath}")


def patch_audio_process(filepath):
    replacements = [
        (
            "def torch_gc():\n    torch.cuda.empty_cache()\n    torch.cuda.ipc_collect()",
            "import sys\nimport os\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))\nfrom ascend_npu_compat import empty_cache, ipc_collect\n\ndef torch_gc():\n    empty_cache()\n    ipc_collect()",
        ),
        (
            "    device='cuda'",
            "    device='npu' if (hasattr(torch, 'npu') and torch.npu.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu')",
        ),
    ]
    patch_file(filepath, replacements)


def patch_run_streamlit(filepath):
    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} not found")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content

    content = content.replace(
        'device = "cuda" if torch.cuda.is_available() else "cpu"',
        'device = "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")',
    )
    content = content.replace(
        "torch.cuda.empty_cache()",
        "empty_cache()",
    )
    content = content.replace(
        "torch.cuda.ipc_collect()",
        "ipc_collect()",
    )

    # Add import
    if "from ascend_npu_compat import empty_cache, ipc_collect" not in content:
        content = content.replace(
            "import streamlit as st",
            "import streamlit as st\nimport sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nfrom ascend_npu_compat import empty_cache, ipc_collect",
        )

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[DONE] Patched {filepath}")
    else:
        print(f"[SKIP] No changes in {filepath}")


def main():
    base = "/opt/atomgit/longcat-video-ascend"

    # Demo scripts
    demos = [
        "run_demo_text_to_video.py",
        "run_demo_image_to_video.py",
        "run_demo_video_continuation.py",
        "run_demo_long_video.py",
        "run_demo_interactive_video.py",
        "run_demo_avatar_single_audio_to_video.py",
        "run_demo_avatar_multi_audio_to_video.py",
    ]
    for demo in demos:
        patch_demo_script(os.path.join(base, demo))

    # Pipelines
    patch_pipeline(os.path.join(base, "longcat_video/pipeline_longcat_video.py"))
    patch_pipeline(os.path.join(base, "longcat_video/pipeline_longcat_video_avatar.py"))

    # Autocast modules
    autocast_files = [
        "longcat_video/modules/longcat_video_dit.py",
        "longcat_video/modules/blocks.py",
        "longcat_video/modules/avatar/longcat_video_dit_avatar.py",
    ]
    for f in autocast_files:
        patch_autocast(os.path.join(base, f))

    # Audio process
    patch_audio_process(os.path.join(base, "longcat_video/audio_process/torch_utils.py"))
    patch_audio_process(os.path.join(base, "longcat_video/audio_process/wav2vec2.py"))

    # Streamlit
    patch_run_streamlit(os.path.join(base, "run_streamlit.py"))

    print("\n=== All patches applied ===")


if __name__ == "__main__":
    main()
