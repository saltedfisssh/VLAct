"""Strict backbone-only initialization from the VLAct continued-pretraining release."""

from pathlib import Path


def select_backbone(checkpoint, expected):
    prefix = "qwen_vl_interface."
    selected = {}
    for key, value in checkpoint.items():
        key = key.removeprefix("module.")
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        # Transformers Qwen3-VL nesting changed across release versions.
        if name.startswith(("model.visual.", "model.language_model.")):
            name = "model.model." + name[len("model.") :]
        if name in selected:
            raise ValueError(f"Duplicate backbone parameter after mapping: {name}")
        selected[name] = value
    missing = sorted(set(expected) - set(selected))
    extra = sorted(set(selected) - set(expected))
    mismatched = [key for key in set(expected) & set(selected) if expected[key].shape != selected[key].shape]
    if missing or extra or mismatched:
        raise ValueError(
            f"Incompatible VLAct backbone: missing={missing[:5]}, extra={extra[:5]}, shapes={mismatched[:5]}"
        )
    return selected


def load_backbone(model, path):
    import torch

    path = Path(path).resolve(strict=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        weights = load_file(str(path))
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    selected = select_backbone(weights, model.qwen_vl_interface.state_dict())
    model.qwen_vl_interface.load_state_dict(selected, strict=True)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "loaded_backbone_tensors": len(selected),
        "loaded_backbone_parameters": sum(v.numel() for v in selected.values()),
        "action_initialization": "Qwen3-0.6B expert plus new PI adapters; source action heads excluded",
    }
