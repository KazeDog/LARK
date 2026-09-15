from __future__ import annotations

from typing import Dict

import torch


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return _strip_module_prefix(checkpoint["model_state_dict"])
    if "state_dict" in checkpoint:
        return _strip_module_prefix(checkpoint["state_dict"])
    return _strip_module_prefix(checkpoint)


def _format_load_result(loaded: bool, result, loaded_keys: int) -> Dict:
    return {
        "loaded": loaded,
        "loaded_keys": int(loaded_keys),
        "missing": list(getattr(result, "missing_keys", [])),
        "unexpected": list(getattr(result, "unexpected_keys", [])),
    }


def freeze_module_parameters(module) -> Dict[str, int]:
    total = 0
    frozen = 0
    for param in module.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            param.requires_grad = False
            frozen += n
    return {"total_params": total, "frozen_params": frozen}


def parameter_trainability_summary(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}


def load_molecular_encoder_weights(model, checkpoint_path: str, map_location: str = "cpu", strict: bool = False) -> Dict:
    """Load a pretrained molecular encoder into a downstream model.

    Downstream models keep the encoder at ``model.backbone.encoder``. Current
    LARK pretraining checkpoints store it under
    ``backbone.molecular_encoder.*``.
    """
    if not checkpoint_path:
        return {"loaded": False, "missing": [], "unexpected": []}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    prefixes = ("backbone.molecular_encoder.", "molecular_encoder.", "encoder.", "backbone.encoder.")
    encoder_state = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                encoder_state[key[len(prefix) :]] = value
                break
    if not encoder_state:
        raise ValueError(f"No molecular encoder weights found in checkpoint: {checkpoint_path}")
    result = model.backbone.encoder.load_state_dict(encoder_state, strict=strict)
    return _format_load_result(True, result, loaded_keys=len(encoder_state))


def load_fusion_backbone_weights(model, checkpoint_path: str, map_location: str = "cpu", strict: bool = False) -> Dict:
    """Load a pretrained FusionBackbone into a downstream reaction model.

    Retrosynthesis keeps the same ``FusionBackbone`` module at ``model.backbone``.
    Pretraining checkpoints store it under ``backbone.*``. Task-specific heads
    are intentionally ignored.
    """

    if not checkpoint_path:
        return {"loaded": False, "loaded_keys": 0, "missing": [], "unexpected": []}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone.") :]] = value
    if not backbone_state:
        raise ValueError(f"No FusionBackbone weights found in checkpoint: {checkpoint_path}")
    result = model.backbone.load_state_dict(backbone_state, strict=strict)
    return _format_load_result(True, result, loaded_keys=len(backbone_state))
