from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from hypermol.models.backbone import FusionBackbone
from hypermol.models.reaction_repr import ReactionRepresentationBackbone, build_encoder_args


BACKBONE_TYPES = {"reaction_repr", "hypergraph"}


class ReactionClassModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_cfg: Dict | None = None,
        reaction_repr_mode: str = "center",
        center_source: str = "auto",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        model_cfg = model_cfg or {}
        requested_backbone = str(model_cfg.get("backbone_type", "reaction_repr")).lower()
        if requested_backbone in {"fusion", "fusion_hypergt", "hypergt", "directed_hypergt"}:
            requested_backbone = "hypergraph"
        if requested_backbone not in BACKBONE_TYPES:
            raise ValueError(f"Unsupported reaction-class backbone_type: {requested_backbone}. Choose from {sorted(BACKBONE_TYPES)}.")
        self.backbone_type = requested_backbone

        if self.backbone_type == "hypergraph":
            self.backbone = FusionBackbone(
                mode="finetune",
                input_num=int(model_cfg.get("input_num", 2)),
                feature=str(model_cfg.get("feature", "hyperedge")),
                mol_embed_dim=int(model_cfg.get("mol_embed_dim", model_cfg.get("embed_dim", 256))),
                mol_num_kernel=int(model_cfg.get("mol_num_kernel", model_cfg.get("num_kernel", 256))),
                mol_num_heads=int(model_cfg.get("mol_num_heads", model_cfg.get("num_heads", 16))),
                mol_num_layers=int(model_cfg.get("mol_num_layers", model_cfg.get("layer_num", 6))),
                mol_hidden_size=int(model_cfg.get("mol_hidden_size", model_cfg.get("hidden_size", 256))),
                hg_embed_dim=int(model_cfg.get("hg_embed_dim", model_cfg.get("mol_embed_dim", 256))),
                hg_num_heads=int(model_cfg.get("hg_num_heads", model_cfg.get("mol_num_heads", 16))),
                hg_layers=int(model_cfg.get("hg_layers", model_cfg.get("mol_num_layers", 6))),
                dropout=dropout,
                num_tasks=int(num_classes),
                condition_enabled=bool(model_cfg.get("condition_enabled", False)),
                condition_dim=int(model_cfg.get("condition_dim", 514)),
                condition_dropout_prob=float(model_cfg.get("condition_dropout_prob", 0.0)),
            )
            output_dim = int(model_cfg.get("hg_embed_dim", model_cfg.get("mol_embed_dim", model_cfg.get("embed_dim", 256))))
        else:
            encoder_args = build_encoder_args({**model_cfg, "dropout": dropout})
            self.backbone = ReactionRepresentationBackbone(
                encoder_args=encoder_args,
                reaction_repr_mode=reaction_repr_mode,
                center_source=center_source,
            )
            output_dim = self.backbone.output_dim
        embed_dim = int(model_cfg.get("mol_embed_dim", model_cfg.get("embed_dim", 256)))
        hidden_dim = int(hidden_dim or model_cfg.get("head_hidden_dim", embed_dim))
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.backbone_type == "hypergraph":
            graph_batch = batch.get("reaction_graph_batch")
            if not graph_batch:
                raise ValueError("ReactionClassModel with backbone_type='hypergraph' requires reaction_graph_batch.")
            features = self.backbone(graph_batch)
            reaction_repr = features["final_edge_features"]
        else:
            features = self.backbone(batch["reactant_batch"], batch["product_batch"])
            reaction_repr = features["reaction_repr"]
        return {"logits": self.head(reaction_repr)}
