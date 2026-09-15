"""BE matrix audit utilities for reaction datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from typing import Iterable

from hypermol.data.reaction_dataset import ReactionDataset
from hypermol.utils.config import load_yaml_config


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def summarize_be_audits(audits: list[dict]) -> dict:
    """Aggregate per-reaction BE audit dictionaries into a compact report."""
    num_samples = len(audits)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "num_bad_samples": 0,
            "num_non_conserved": 0,
            "num_map_mismatch": 0,
            "num_negative_diagonal": 0,
        }

    def count_where(key, predicate=bool):
        return sum(1 for audit in audits if predicate(audit.get(key)))

    num_negative_diagonal = sum(
        1
        for audit in audits
        if audit.get("reactant_negative_diagonal", 0) > 0 or audit.get("product_negative_diagonal", 0) > 0
    )
    num_map_mismatch = sum(
        1
        for audit in audits
        if audit.get("num_reactant_only_maps", 0) > 0 or audit.get("num_product_only_maps", 0) > 0
    )
    bad_counter = Counter()
    for audit in audits:
        if not audit.get("electron_conserved", True):
            bad_counter["non_conserved"] += 1
        if not audit.get("reactant_symmetric", True) or not audit.get("product_symmetric", True):
            bad_counter["not_symmetric"] += 1
        if not audit.get("delta_symmetric", True):
            bad_counter["delta_not_symmetric"] += 1
        if audit.get("duplicate_reactant_maps", 0) > 0 or audit.get("duplicate_product_maps", 0) > 0:
            bad_counter["duplicate_maps"] += 1
        if audit.get("num_reactant_only_maps", 0) > 0 or audit.get("num_product_only_maps", 0) > 0:
            bad_counter["map_mismatch"] += 1
        if audit.get("reactant_negative_diagonal", 0) > 0 or audit.get("product_negative_diagonal", 0) > 0:
            bad_counter["negative_diagonal"] += 1

    return {
        "num_samples": num_samples,
        "num_bad_samples": sum(1 for audit in audits if is_bad_be_audit_dict(audit)),
        "num_non_conserved": count_where("electron_conserved", lambda value: not bool(value)),
        "num_map_mismatch": num_map_mismatch,
        "num_negative_diagonal": num_negative_diagonal,
        "num_duplicate_map_samples": sum(
            1
            for audit in audits
            if audit.get("duplicate_reactant_maps", 0) > 0 or audit.get("duplicate_product_maps", 0) > 0
        ),
        "num_reactant_not_symmetric": count_where("reactant_symmetric", lambda value: not bool(value)),
        "num_product_not_symmetric": count_where("product_symmetric", lambda value: not bool(value)),
        "num_delta_not_symmetric": count_where("delta_symmetric", lambda value: not bool(value)),
        "mean_abs_delta_sum": _mean(abs(float(audit.get("delta_sum", 0.0))) for audit in audits),
        "max_abs_delta_sum": max(abs(float(audit.get("delta_sum", 0.0))) for audit in audits),
        "mean_max_abs_delta": _mean(float(audit.get("max_abs_delta", 0.0)) for audit in audits),
        "issue_counts": dict(bad_counter),
    }


def is_bad_be_audit_dict(audit: dict) -> bool:
    """Dict equivalent of ``is_bad_be_audit`` for serialized audit reports."""
    return (
        not audit.get("electron_conserved", True)
        or not audit.get("reactant_symmetric", True)
        or not audit.get("product_symmetric", True)
        or not audit.get("delta_symmetric", True)
        or audit.get("reactant_negative_diagonal", 0) > 0
        or audit.get("product_negative_diagonal", 0) > 0
        or audit.get("duplicate_reactant_maps", 0) > 0
        or audit.get("duplicate_product_maps", 0) > 0
        or audit.get("num_reactant_only_maps", 0) > 0
        or audit.get("num_product_only_maps", 0) > 0
    )


def audit_reaction_dataset(dataset: ReactionDataset, max_samples: int = 0) -> dict:
    """Run BE audits over an instantiated ``ReactionDataset``."""
    audits = []
    examples = []
    limit = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    for idx in range(limit):
        sample = dataset[idx]
        if not sample or "be_audit" not in sample:
            continue
        audit = sample["be_audit"]
        audits.append(audit)
        if len(examples) < 20 and is_bad_be_audit_dict(audit):
            examples.append(
                {
                    "index": idx,
                    "reactants": sample.get("raw_reactant_smiles", []),
                    "products": sample.get("raw_product_smiles", []),
                    "audit": audit,
                }
            )
    return {"summary": summarize_be_audits(audits), "examples": examples}


def audit_reaction_dataset_from_paths(
    split_csv: str,
    molecule_db: str,
    aromatic_mode: str = "aromatic_1p5",
    max_samples: int = 0,
    tolerance: float = 1e-4,
) -> dict:
    """Build a dataset from paths and return its BE audit report."""
    dataset = ReactionDataset(
        split_csv=split_csv,
        molecule_db=molecule_db,
        mode="finetune",
        task=None,
        aromatic_mode=aromatic_mode,
        be_audit_tolerance=tolerance,
        return_be_audit=True,
    )
    return audit_reaction_dataset(dataset, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reaction BE/Delta-BE matrix quality.")
    parser.add_argument("--config", default="")
    parser.add_argument("--split_csv", default="")
    parser.add_argument("--molecule_db", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--aromatic_mode", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config).raw
        data_root = cfg["data_root"]
        db_name = cfg.get("molecule_db", "smiles.lmdb")
        split_csv = os.path.join(data_root, f"{args.split}.csv")
        molecule_db = db_name if os.path.isabs(db_name) else os.path.join(data_root, db_name)
        be_cfg = cfg.get("be_matrix", {})
        audit_cfg = be_cfg.get("audit", {})
        aromatic_mode = args.aromatic_mode or be_cfg.get("aromatic_mode", "aromatic_1p5")
        max_samples = args.max_samples or int(audit_cfg.get("max_samples", 0) or 0)
        tolerance = args.tolerance if args.tolerance is not None else float(audit_cfg.get("tolerance", 1e-4))
    else:
        if not args.split_csv or not args.molecule_db:
            raise ValueError("Provide either --config or both --split_csv and --molecule_db.")
        split_csv = args.split_csv
        molecule_db = args.molecule_db
        aromatic_mode = args.aromatic_mode or "aromatic_1p5"
        max_samples = args.max_samples
        tolerance = args.tolerance

    report = audit_reaction_dataset_from_paths(
        split_csv=split_csv,
        molecule_db=molecule_db,
        aromatic_mode=aromatic_mode,
        max_samples=max_samples,
        tolerance=tolerance,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
