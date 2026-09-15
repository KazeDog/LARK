"""Training budget and resume validation."""
from typing import Any, Mapping

def resolve_training_budget(config: Mapping[str, Any], *, world_size: int) -> dict[str, Any]:
    """Normalize and validate the optional global pretraining budget."""

    raw = config.get("training_budget") or {}
    if not raw:
        return {
            "enabled": False,
            "max_optimizer_steps": 0,
            "expected_world_size": int(world_size),
            "view_multiplier": 1,
            "local_graph_views_per_step": 0,
            "graph_views_per_optimizer_step": 0,
            "max_graph_views": 0,
            "require_full_train_batches": False,
            "enforce_runtime_graph_views_per_step": False,
            "require_exact_budget": False,
            "checkpoint_selection": "best_validation",
        }
    if not isinstance(raw, Mapping):
        raise TypeError("training_budget must be a mapping.")
    world_size = int(world_size)
    expected_world_size = int(raw.get("expected_world_size", world_size))
    if expected_world_size != world_size:
        raise ValueError(
            "Training-budget world size mismatch: "
            f"configured={expected_world_size}, runtime={world_size}."
        )
    mode = str(
        (config.get("reaction_objective") or {}).get("mode", "reaction_reconstruction")
    ).lower()
    inferred_multiplier = 2 if mode in {"bidirectional", "bidirectional_single_side"} else 1
    view_multiplier = int(raw.get("view_multiplier", inferred_multiplier))
    if view_multiplier != inferred_multiplier:
        raise ValueError(
            "training_budget.view_multiplier disagrees with reaction_objective.mode: "
            f"configured={view_multiplier}, inferred={inferred_multiplier}."
        )
    max_steps = int(raw.get("max_optimizer_steps", 0))
    configured_views = int(raw.get("graph_views_per_optimizer_step", 0))
    actual_views = int(config["batch_size"]) * world_size * view_multiplier
    if max_steps <= 0 or configured_views <= 0:
        raise ValueError(
            "Formal training_budget requires positive max_optimizer_steps and "
            "graph_views_per_optimizer_step."
        )
    if configured_views != actual_views:
        raise ValueError(
            "Graph views per optimizer step do not match runtime batch geometry: "
            f"configured={configured_views}, actual={actual_views}."
        )
    configured_max_views = int(raw.get("max_graph_views", max_steps * configured_views))
    if configured_max_views != max_steps * configured_views:
        raise ValueError("training_budget.max_graph_views must equal steps * graph views per step.")
    require_exact = bool(raw.get("require_exact_budget", True))
    checkpoint_selection = str(
        raw.get("checkpoint_selection", "final_budget" if require_exact else "best_validation")
    ).strip().lower()
    if checkpoint_selection not in {"best_validation", "final_budget"}:
        raise ValueError(
            "training_budget.checkpoint_selection must be 'best_validation' or 'final_budget'."
        )
    if checkpoint_selection == "final_budget" and not require_exact:
        raise ValueError("checkpoint_selection='final_budget' requires require_exact_budget=true.")
    if require_exact and int(config.get("early_stop_patience", 0) or 0) > 0:
        raise ValueError("Exact training budgets require early_stop_patience=0.")
    return {
        "enabled": True,
        "schema_version": int(raw.get("schema_version", 1)),
        "unit": str(raw.get("unit", "optimizer_step_with_matched_graph_views")),
        "max_optimizer_steps": max_steps,
        "expected_world_size": expected_world_size,
        "view_multiplier": view_multiplier,
        "local_graph_views_per_step": int(config["batch_size"]) * view_multiplier,
        "graph_views_per_optimizer_step": configured_views,
        "max_graph_views": configured_max_views,
        "require_full_train_batches": bool(raw.get("require_full_train_batches", False)),
        "enforce_runtime_graph_views_per_step": bool(
            raw.get("enforce_runtime_graph_views_per_step", False)
        ),
        "require_exact_budget": require_exact,
        "checkpoint_selection": checkpoint_selection,
    }

def validate_resume_training_budget(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    runtime = resolve_training_budget(config, world_size=world_size)
    saved_cfg = checkpoint.get("config") or {}
    saved_runtime = resolve_training_budget(saved_cfg, world_size=world_size)
    comparable_keys = (
        "enabled",
        "max_optimizer_steps",
        "expected_world_size",
        "view_multiplier",
        "graph_views_per_optimizer_step",
        "max_graph_views",
        "require_full_train_batches",
        "enforce_runtime_graph_views_per_step",
        "require_exact_budget",
        "checkpoint_selection",
    )
    expected = {key: runtime[key] for key in comparable_keys}
    observed = {key: saved_runtime[key] for key in comparable_keys}
    if expected != observed:
        raise ValueError(
            "Cannot resume with a different training budget: "
            f"config={expected!r}, checkpoint={observed!r}."
        )
    if runtime["enabled"] and int(checkpoint.get("epoch", 0)) > 0:
        if "global_optimizer_step" not in checkpoint:
            raise ValueError(
                "A budgeted resume requires checkpoint.global_optimizer_step; "
                "use --init_checkpoint for historical checkpoints."
            )
    return runtime
