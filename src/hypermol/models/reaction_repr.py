from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hypermol.data.preprocess import CompoundKit
from hypermol.models.molecular_encoder import MolecularEncoder


REACTION_REPR_MODES = {"concat", "diff", "rich", "center"}


def build_encoder_args(model_cfg: Dict) -> Dict:
    return {
        "mode": "finetune",
        "atom_names": list(CompoundKit.atom_vocab_dict.keys()),
        "bond_names": list(CompoundKit.bond_vocab_dict.keys()),
        "embed_dim": int(model_cfg.get("mol_embed_dim", model_cfg.get("embed_dim", 256))),
        "num_kernel": int(model_cfg.get("mol_num_kernel", model_cfg.get("num_kernel", 256))),
        "layer_num": int(model_cfg.get("mol_num_layers", model_cfg.get("layer_num", 6))),
        "num_heads": int(model_cfg.get("mol_num_heads", model_cfg.get("num_heads", 16))),
        "hidden_size": int(model_cfg.get("mol_hidden_size", model_cfg.get("hidden_size", 256))),
        "cross_layers": int(model_cfg.get("cross_layers", 100)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }


class ReactionRepresentationBackbone(nn.Module):
    """Shared reactant/product encoder for downstream reaction tasks."""

    def __init__(
        self,
        encoder_args: Dict,
        reaction_repr_mode: str = "rich",
        center_source: str = "auto",
    ):
        super().__init__()
        if reaction_repr_mode not in REACTION_REPR_MODES:
            raise ValueError(f"Unsupported reaction_repr_mode: {reaction_repr_mode}")
        if center_source not in {"auto", "real", "fallback"}:
            raise ValueError(f"Unsupported center_source: {center_source}")
        self.encoder = MolecularEncoder(**encoder_args)
        self.reaction_repr_mode = reaction_repr_mode
        self.center_source = center_source
        self.embed_dim = int(encoder_args["embed_dim"])
        self.center_query_proj = nn.Linear(self.embed_dim, self.embed_dim) if reaction_repr_mode == "center" else None

    @property
    def output_dim(self) -> int:
        if self.reaction_repr_mode == "concat":
            return self.embed_dim * 2
        if self.reaction_repr_mode == "diff":
            return self.embed_dim
        if self.reaction_repr_mode == "rich":
            return self.embed_dim * 4
        return self.embed_dim * 8

    def _build_graph_repr(self, reactant_feat: torch.Tensor, product_feat: torch.Tensor) -> torch.Tensor:
        if self.reaction_repr_mode == "concat":
            return torch.cat([reactant_feat, product_feat], dim=-1)
        if self.reaction_repr_mode == "diff":
            return product_feat - reactant_feat
        return torch.cat(
            [reactant_feat, product_feat, product_feat - reactant_feat, product_feat * reactant_feat],
            dim=-1,
        )

    def _pool_by_query(self, atom_features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if atom_features.shape[0] == 0:
            return torch.zeros_like(query)
        projected_query = self.center_query_proj(query)
        scores = torch.matmul(atom_features, projected_query) / (float(atom_features.shape[-1]) ** 0.5)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(atom_features * weights.unsqueeze(-1), dim=0)

    @staticmethod
    def _pool_optional_subset(atom_features: torch.Tensor, indices: List[int]) -> Optional[torch.Tensor]:
        if not indices:
            return None
        return atom_features[torch.as_tensor(indices, device=atom_features.device, dtype=torch.long)].mean(dim=0)

    def _pool_mapped_center(
        self,
        reactant_atom_features: torch.Tensor,
        product_atom_features: torch.Tensor,
        reactant_map_num: torch.Tensor,
        product_map_num: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        reactant_map_to_idx = {
            int(map_num): idx for idx, map_num in enumerate(reactant_map_num.tolist()) if int(map_num) > 0
        }
        product_map_to_idx = {
            int(map_num): idx for idx, map_num in enumerate(product_map_num.tolist()) if int(map_num) > 0
        }
        shared_map_nums = sorted(set(reactant_map_to_idx) & set(product_map_to_idx))
        if not shared_map_nums:
            return None

        reactant_shared = torch.stack(
            [reactant_atom_features[reactant_map_to_idx[map_num]] for map_num in shared_map_nums], dim=0
        )
        product_shared = torch.stack(
            [product_atom_features[product_map_to_idx[map_num]] for map_num in shared_map_nums], dim=0
        )
        delta = product_shared - reactant_shared
        scores = torch.norm(delta, dim=-1)
        weights = torch.softmax(scores, dim=0)
        center_r = torch.sum(reactant_shared * weights.unsqueeze(-1), dim=0)
        center_p = torch.sum(product_shared * weights.unsqueeze(-1), dim=0)

        reactant_only = [idx for map_num, idx in reactant_map_to_idx.items() if map_num not in product_map_to_idx]
        product_only = [idx for map_num, idx in product_map_to_idx.items() if map_num not in reactant_map_to_idx]
        reactant_extra = self._pool_optional_subset(reactant_atom_features, reactant_only)
        product_extra = self._pool_optional_subset(product_atom_features, product_only)
        if reactant_extra is not None:
            center_r = torch.stack([center_r, reactant_extra], dim=0).mean(dim=0)
        if product_extra is not None:
            center_p = torch.stack([center_p, product_extra], dim=0).mean(dim=0)
        return center_r, center_p

    def _build_center_repr(
        self,
        reactant_outputs: Dict[str, torch.Tensor],
        product_outputs: Dict[str, torch.Tensor],
        reactant_batch: Dict[str, torch.Tensor],
        product_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        reactant_atoms = reactant_outputs["atom_features"]
        product_atoms = product_outputs["atom_features"]
        reactant_graph = reactant_outputs["mol_features"]
        product_graph = product_outputs["mol_features"]
        reactant_mask = reactant_batch.get("atom_mask")
        product_mask = product_batch.get("atom_mask")
        reactant_map_num = reactant_batch.get("atom_map_num")
        product_map_num = product_batch.get("atom_map_num")

        center_features: List[torch.Tensor] = []
        for i in range(reactant_atoms.shape[0]):
            reactant_valid = reactant_mask[i].bool() if reactant_mask is not None else torch.ones(
                reactant_atoms.shape[1], dtype=torch.bool, device=reactant_atoms.device
            )
            product_valid = product_mask[i].bool() if product_mask is not None else torch.ones(
                product_atoms.shape[1], dtype=torch.bool, device=product_atoms.device
            )
            reactant_atom_features = reactant_atoms[i, reactant_valid]
            product_atom_features = product_atoms[i, product_valid]
            query = product_graph[i] - reactant_graph[i]

            center_pair = None
            use_real_center = self.center_source in {"auto", "real"}
            if use_real_center and reactant_map_num is not None and product_map_num is not None:
                reactant_maps = reactant_map_num[i, reactant_valid]
                product_maps = product_map_num[i, product_valid]
                if torch.any(reactant_maps > 0) and torch.any(product_maps > 0):
                    center_pair = self._pool_mapped_center(
                        reactant_atom_features,
                        product_atom_features,
                        reactant_maps,
                        product_maps,
                    )

            if center_pair is None:
                center_r = self._pool_by_query(reactant_atom_features, query)
                center_p = self._pool_by_query(product_atom_features, query)
            else:
                center_r, center_p = center_pair

            center_features.append(torch.cat([center_r, center_p, center_p - center_r, center_p * center_r], dim=-1))
        return torch.stack(center_features, dim=0)

    def forward(self, reactant_batch: Dict[str, torch.Tensor], product_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reactant_outputs = self.encoder(reactant_batch)
        product_outputs = self.encoder(product_batch)
        reactant_feat = reactant_outputs["mol_features"]
        product_feat = product_outputs["mol_features"]
        reaction_repr = self._build_graph_repr(reactant_feat, product_feat)
        if self.reaction_repr_mode == "center":
            center_repr = self._build_center_repr(reactant_outputs, product_outputs, reactant_batch, product_batch)
            reaction_repr = torch.cat([reaction_repr, center_repr], dim=-1)
        return {
            "reaction_repr": reaction_repr,
            "reactant_outputs": reactant_outputs,
            "product_outputs": product_outputs,
        }
