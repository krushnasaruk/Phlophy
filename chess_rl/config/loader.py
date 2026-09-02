"""
Configuration loader and validator for chess_rl.
Handles YAML parsing, sub-config reference resolution, type coercion, and CLI overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from chess_rl.config.schema import (
    ExperimentConfig,
    ExperimentMetaConfig,
    EnvironmentConfig,
    RulesConfig,
    RewardSchemeConfig,
    ModelConfig,
    TrainingConfig,
    MCTSConfig,
    ReplayBufferConfig,
    EvaluationConfig,
    CheckpointingConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    LossWeightsConfig,
    CurriculumConfig,
    CurriculumStageConfig,
)


def _load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_sub_config(
    base_dir: Path, sub_dict_or_path: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve a nested config if specified as a file path or return dict."""
    if isinstance(sub_dict_or_path, str):
        ref_path = base_dir / sub_dict_or_path
        if not ref_path.exists():
            # Try from current working directory
            ref_path = Path(sub_dict_or_path)
        return _load_yaml_file(ref_path)
    elif isinstance(sub_dict_or_path, dict):
        if "config_path" in sub_dict_or_path:
            ref_path = base_dir / sub_dict_or_path["config_path"]
            if not ref_path.exists():
                ref_path = Path(sub_dict_or_path["config_path"])
            loaded = _load_yaml_file(ref_path)
            # Merge any inline overrides
            for k, v in sub_dict_or_path.items():
                if k != "config_path":
                    loaded[k] = v
            return loaded
        return sub_dict_or_path
    return {}


def _apply_dict_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply dot-notation overrides into a nested dictionary."""
    for key, value in overrides.items():
        parts = key.split(".")
        current = target
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


def parse_environment_config(data: Dict[str, Any]) -> EnvironmentConfig:
    rules_data = data.get("rules", {})
    rules_cfg = RulesConfig(**{k: v for k, v in rules_data.items() if hasattr(RulesConfig, k)})

    reward_data = data.get("reward_scheme", {})
    reward_cfg = RewardSchemeConfig(
        **{k: v for k, v in reward_data.items() if hasattr(RewardSchemeConfig, k)}
    )

    env_kwargs = {
        k: v
        for k, v in data.items()
        if hasattr(EnvironmentConfig, k) and k not in ["rules", "reward_scheme"]
    }
    return EnvironmentConfig(rules=rules_cfg, reward_scheme=reward_cfg, **env_kwargs)


def parse_model_config(data: Dict[str, Any]) -> ModelConfig:
    kwargs = {k: v for k, v in data.items() if hasattr(ModelConfig, k)}
    return ModelConfig(**kwargs)


def parse_training_config(data: Dict[str, Any]) -> TrainingConfig:
    mcts_data = data.get("mcts", {})
    mcts_cfg = MCTSConfig(**{k: v for k, v in mcts_data.items() if hasattr(MCTSConfig, k)})

    replay_data = data.get("replay_buffer", {})
    replay_cfg = ReplayBufferConfig(
        **{k: v for k, v in replay_data.items() if hasattr(ReplayBufferConfig, k)}
    )

    eval_data = data.get("evaluation", {})
    eval_cfg = EvaluationConfig(
        **{k: v for k, v in eval_data.items() if hasattr(EvaluationConfig, k)}
    )

    ckpt_data = data.get("checkpointing", {})
    ckpt_cfg = CheckpointingConfig(
        **{k: v for k, v in ckpt_data.items() if hasattr(CheckpointingConfig, k)}
    )

    training_kwargs = {
        k: v
        for k, v in data.items()
        if hasattr(TrainingConfig, k)
        and k not in ["mcts", "replay_buffer", "evaluation", "checkpointing"]
    }
    return TrainingConfig(
        mcts=mcts_cfg,
        replay_buffer=replay_cfg,
        evaluation=eval_cfg,
        checkpointing=ckpt_cfg,
        **training_kwargs,
    )


def parse_optimizer_config(data: Dict[str, Any]) -> OptimizerConfig:
    lr_data = data.get("lr_scheduler", {})
    lr_cfg = LRSchedulerConfig(**{k: v for k, v in lr_data.items() if hasattr(LRSchedulerConfig, k)})

    loss_data = data.get("loss_weights", {})
    loss_cfg = LossWeightsConfig(
        **{k: v for k, v in loss_data.items() if hasattr(LossWeightsConfig, k)}
    )

    opt_kwargs = {
        k: v
        for k, v in data.items()
        if hasattr(OptimizerConfig, k) and k not in ["lr_scheduler", "loss_weights"]
    }
    return OptimizerConfig(lr_scheduler=lr_cfg, loss_weights=loss_cfg, **opt_kwargs)


def parse_curriculum_config(data: Dict[str, Any]) -> CurriculumConfig:
    stages_data = data.get("stages", [])
    stages = []
    for s_data in stages_data:
        stage_kwargs = {k: v for k, v in s_data.items() if hasattr(CurriculumStageConfig, k)}
        stages.append(CurriculumStageConfig(**stage_kwargs))

    curriculum_kwargs = {
        k: v for k, v in data.items() if hasattr(CurriculumConfig, k) and k != "stages"
    }
    return CurriculumConfig(stages=stages, **curriculum_kwargs)


def load_experiment_config(
    config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """
    Load an experiment configuration YAML and its nested sub-configurations.

    Args:
        config_path: Path to the main experiment YAML configuration file.
        overrides: Optional dictionary of dot-notated overrides (e.g. {'experiment.seed': 100}).

    Returns:
        Fully initialized and typed ExperimentConfig.
    """
    path = Path(config_path)
    base_dir = path.parent
    raw_exp = _load_yaml_file(path)

    if overrides:
        _apply_dict_overrides(raw_exp, overrides)

    # Resolve sub-configs
    exp_meta_dict = raw_exp.get("experiment", {})
    exp_meta = ExperimentMetaConfig(
        **{k: v for k, v in exp_meta_dict.items() if hasattr(ExperimentMetaConfig, k)}
    )

    env_dict = _resolve_sub_config(base_dir, raw_exp.get("environment", {}))
    env_cfg = parse_environment_config(env_dict)

    model_dict = _resolve_sub_config(base_dir, raw_exp.get("model", {}))
    model_cfg = parse_model_config(model_dict)

    train_dict = _resolve_sub_config(base_dir, raw_exp.get("training", {}))
    train_cfg = parse_training_config(train_dict)

    opt_dict = _resolve_sub_config(base_dir, raw_exp.get("optimizer", {}))
    opt_cfg = parse_optimizer_config(opt_dict)

    curr_dict = raw_exp.get("curriculum", {})
    curr_cfg = parse_curriculum_config(curr_dict)

    return ExperimentConfig(
        experiment=exp_meta,
        environment=env_cfg,
        model=model_cfg,
        training=train_cfg,
        optimizer=opt_cfg,
        curriculum=curr_cfg,
        raw_config=raw_exp,
    )
