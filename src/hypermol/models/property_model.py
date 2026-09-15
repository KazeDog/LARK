"""Single-molecule masked multi-task property prediction with LARK."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from hypermol.models.molecular_encoder import MolecularEncoder
from hypermol.models.reaction_repr import build_encoder_args


class MolecularPropertyBackbone(nn.Module):
    """Thin wrapper that preserves the shared ``model.backbone.encoder`` path."""

    def __init__(self, model_cfg: Dict, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else MolecularEncoder(**build_encoder_args(model_cfg))

    def forward(self, molecule_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encoder(molecule_batch)


class MolecularPropertyModel(nn.Module):
    """Predict one or more property outputs from a graph-token feature."""

    def __init__(
        self,
        model_cfg: Dict | None = None,
        hidden_dim: int | None = None,
        dropout: float | None = None,
        freeze_encoder: bool = False,
        encoder: nn.Module | None = None,
        num_tasks: int = 1,
    ):
        super().__init__()
        model_cfg = dict(model_cfg or {})
        if dropout is not None:
            model_cfg["dropout"] = float(dropout)
        dropout = float(model_cfg.get("dropout", 0.1))
        embed_dim = int(model_cfg.get("mol_embed_dim", model_cfg.get("embed_dim", 256)))
        hidden_dim = int(hidden_dim or model_cfg.get("head_hidden_dim", embed_dim))
        self.num_tasks = int(num_tasks)
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")

        self.backbone = MolecularPropertyBackbone(model_cfg, encoder=encoder)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_tasks),
        )
        self.encoder_frozen = False
        self.set_encoder_frozen(freeze_encoder)

    def set_encoder_frozen(self, frozen: bool = True) -> None:
        """Freeze/unfreeze the encoder and enforce the appropriate module mode."""

        self.encoder_frozen = bool(frozen)
        for parameter in self.backbone.encoder.parameters():
            parameter.requires_grad = not self.encoder_frozen
        if self.encoder_frozen:
            self.backbone.encoder.eval()
        else:
            self.backbone.encoder.train(self.training)

    def train(self, mode: bool = True):
        """Keep a frozen encoder deterministic when the task head is training."""

        super().train(mode)
        encoder_parameters = list(self.backbone.encoder.parameters())
        externally_frozen = bool(encoder_parameters) and not any(
            parameter.requires_grad for parameter in encoder_parameters
        )
        if self.encoder_frozen or externally_frozen:
            self.backbone.encoder.eval()
        return self

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        molecule_batch = batch.get("molecule_batch")
        if not molecule_batch:
            raise ValueError("MolecularPropertyModel requires a non-empty molecule_batch.")
        encoder_outputs = self.backbone(molecule_batch)
        mol_features = encoder_outputs["mol_features"]
        logits = self.head(mol_features)
        if self.num_tasks == 1:
            logits = logits.squeeze(-1)
        return {
            "logits": logits,
            "mol_features": mol_features,
        }


__all__ = ["MolecularPropertyBackbone", "MolecularPropertyModel"]
