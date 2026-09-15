import torch.nn.functional as F
from torch import nn

from hypermol.models.backbone import FusionBackbone
from hypermol.models.heads import MolProjection, AngleProjection, TorsionHead, RMatrixHead, RMatrixHeadGAT
from hypermol.utils.pretrain_tasks import get_model_required_tasks, normalize_pretrain_tasks


class PretrainModel(nn.Module):
    def __init__(self, mode: str = 'pretrain',
                 input_num = 2,
                 feature = 'hyperedge',
                 # 分子编码器参数
                 mol_embed_dim: int = 256,
                 mol_num_kernel = 256,
                 mol_num_heads: int = 16,
                 mol_num_layers: int = 6,
                 mol_hidden_size: int = 256,
                 # 超图网络参数
	                 hg_embed_dim: int = 256,
	                 hg_num_heads: int = 16,
	                 hg_layers: int = 6,
	                 # 通用参数
	                 dropout: float = 0.1,
	                 r_matrix_use_be_pair_feature: bool = False,
	                 be_rbf_bins: int = 81,
	                 be_rbf_min: float = 0.0,
	                 be_rbf_max: float = 8.0,
	                 condition_enabled: bool = False,
	                 condition_dim: int = 514,
	                 condition_dropout_prob: float = 0.0,
	                 context_role_enabled: bool = False,
	                 consistency_enabled: bool = False,
	                 consistency_projection_dim: int = 128,
	                 active_tasks=None,
	                 num_tasks=1,
	                 **kwargs):
        super().__init__()

        self.mode = mode
        self.input_num = input_num
        self.feature = feature

        self.mol_embed_dim = mol_embed_dim
        self.mol_num_kernel = mol_num_kernel
        self.mol_num_heads = mol_num_heads
        self.mol_num_layers = mol_num_layers
        self.mol_hidden_size = mol_hidden_size

        self.hg_embed_dim = hg_embed_dim
        self.hg_num_heads = hg_num_heads
        self.hg_layers = hg_layers

        self.dropout = dropout
        self.num_tasks = num_tasks
        self.active_tasks = set(normalize_pretrain_tasks(active_tasks))
        self.model_tasks = set(get_model_required_tasks(active_tasks))
        self.consistency_enabled = bool(consistency_enabled)
        self.consistency_projection_dim = int(consistency_projection_dim)
        if self.consistency_enabled and self.consistency_projection_dim <= 0:
            raise ValueError("consistency_projection_dim must be > 0 when consistency is enabled.")

        self.backbone = FusionBackbone(
            self.mode,
            self.input_num,
            self.feature,
            self.mol_embed_dim,
            self.mol_num_kernel,
            self.mol_num_heads,
            self.mol_num_layers,
            self.mol_hidden_size,
            self.hg_embed_dim,
            self.hg_num_heads,
            self.hg_layers,
            self.dropout,
            self.num_tasks,
            condition_enabled=condition_enabled,
            condition_dim=condition_dim,
            condition_dropout_prob=condition_dropout_prob,
            context_role_enabled=context_role_enabled,
        )

        self.consistency_projector = (
            nn.Sequential(
                nn.Linear(self.hg_embed_dim, self.hg_embed_dim),
                nn.LayerNorm(self.hg_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hg_embed_dim, self.consistency_projection_dim),
            )
            if self.consistency_enabled
            else None
        )

        self.mam_head = MolProjection(self.mol_embed_dim, self.mol_embed_dim // 2, 122, dropout=self.dropout)

        self.angle_head = AngleProjection(self.mol_embed_dim, self.mol_embed_dim // 2, 20, dropout=self.dropout)

        self.torsion_head = TorsionHead(self.mol_embed_dim, self.mol_embed_dim // 2, 36, dropout=self.dropout)

        self.fingerprint_head = nn.Sequential(
            MolProjection(self.mol_embed_dim, self.mol_embed_dim * 2, 2048 * 2 + 166, dropout=self.dropout),
        )

        self.r_matrix_head = RMatrixHead(
            self.mol_embed_dim,
            dropout=self.dropout,
            use_be_pair_feature=r_matrix_use_be_pair_feature,
            be_rbf_bins=be_rbf_bins,
            be_rbf_min=be_rbf_min,
            be_rbf_max=be_rbf_max,
        )
        # self.r_matrix_head = RMatrixHeadGAT(self.mol_embed_dim, dropout=self.dropout)

    def forward(self, batch):
        features = self.backbone(batch)
        context_aware_atom_embeddings = features["context_aware_atom_embeddings"]

        predictions = {}
        if self.consistency_projector is not None:
            predictions["consistency_embeddings"] = F.normalize(
                self.consistency_projector(features["core_view_features"]),
                p=2,
                dim=-1,
            )
        # atom_feats = features["context_aware_atom_embeddings"]
        if "mam" in self.model_tasks:
            mam_mask = batch['mam_loss_mask']
            # masked_atom_reps = dense_atom_embeddings[mam_mask]
            masked_atom_reps = context_aware_atom_embeddings[mam_mask]
            if masked_atom_reps.shape[0] > 0:
                predictions["mam_logits"] = self.mam_head(masked_atom_reps)

        if "angle" in self.model_tasks:
            predictions["angle_logits"] = self.angle_head(
                context_aware_atom_embeddings,
                batch['angles_atom_index'],
                batch['angle_valid_mask']
            )

        if "torsion" in self.model_tasks and 'edges' in batch and 'torsion_valid_mask' in batch:
            predictions["torsion_logits"] = self.torsion_head(
                context_aware_atom_embeddings,
                batch['edges'],
                batch['torsion_valid_mask']
            )

        if "fingerprint" in self.model_tasks:
            predictions["fingerprint_logits"] = self.fingerprint_head(features['initial_node_embeddings'])

        # b. R矩阵预测
        if "r_matrix" in self.model_tasks:
            observed_mask = batch.get(
                "observed_mask",
                batch.get("observed_pair_mask", batch.get("pair_prior_observed_mask")),
            )
            r_matrix_preds, _ = self.r_matrix_head(
                context_aware_atom_embeddings,
                batch['atom_mask'],
                batch.get('map_lists', []),
                batch['batch_vec'],
                batch['reaction_canvas_mask'],
                batch.get('padded_be_matrix'),
                pair_prior=batch.get('pair_prior'),
                observed_mask=observed_mask,
                canvas_index_lists=batch.get('canvas_index_lists'),
            )

            # r_matrix_preds, _ = self.r_matrix_head(
            #     context_aware_atom_embeddings,
            #     batch['atom_mask'],
            #     batch['map_lists'],
            #     batch['batch_vec'],
            #     batch['reaction_canvas_mask'],
            #     batch['padded_be_matrix'],
            # )
            # Keep the historical key for all existing training/evaluation
            # code and expose the chemically explicit name for strict
            # Delta-BE pipelines.  Both names intentionally reference the same
            # tensor, so there is no duplicated compute or checkpoint state.
            predictions['r_matrix_preds'] = r_matrix_preds
            predictions['delta_be_preds'] = r_matrix_preds

        return predictions
