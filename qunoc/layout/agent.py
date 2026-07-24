import os
import tempfile
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor

from ..element.package import Package
from .env import NoFeasibleDesignError, QuNoCEnv
from .schema import (
    build_model_manifest,
    validate_model_manifest,
    write_model_manifest,
)


class LayoutFeaturesExtractor(BaseFeaturesExtractor):
    """Fixed layout/context encoder shared by every fixed-schema environment."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        layout_space = observation_space.spaces["layout"]
        context_space = observation_space.spaces["context"]
        in_channels = layout_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 192),
            nn.ReLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(192 + 64, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        layout_features = self.cnn(observations["layout"].float())
        context_features = self.context_mlp(observations["context"].float())
        return self.fusion(torch.cat((layout_features, context_features), dim=1))


class LayoutPolicy(ActorCriticPolicy):
    """Actor-critic policy with environment-provided joint action masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=LayoutFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            **kwargs,
        )
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise TypeError("LayoutPolicy requires a fixed Discrete action space.")

    def _masked_distribution(
        self, latent_pi: torch.Tensor, action_masks: torch.Tensor
    ):
        logits = self.action_net(latent_pi)
        masks = action_masks.to(device=logits.device, dtype=torch.bool)
        if not masks.any(dim=1).all():
            raise RuntimeError("Every layout state must expose at least one legal action.")
        masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._masked_distribution(latent_pi, obs["action_mask"])
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._masked_distribution(latent_pi, obs["action_mask"])
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions.flatten())
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._masked_distribution(latent_pi, obs["action_mask"])


class RolloutProgressCallback(BaseCallback):
    """Print rollout progress and layout action quality metrics."""

    def __init__(self):
        super().__init__()
        self.progress_interval = 8
        self.rollout_index = 0
        self.rollout_steps = 0
        self.rollout_target_steps = 0
        self.rollout_episodes = 0
        self.rollout_action_count = 0
        self.rollout_invalid_actions = 0
        self.termination_reasons = Counter()
        self.last_valid_action_count = None

    def _on_rollout_start(self) -> None:
        self.rollout_index += 1
        self.rollout_steps = 0
        self.rollout_episodes = 0
        self.rollout_action_count = 0
        self.rollout_invalid_actions = 0
        self.termination_reasons.clear()
        self.last_valid_action_count = None
        num_envs = getattr(self.training_env, "num_envs", 1)
        self.rollout_target_steps = self.model.n_steps * num_envs
        print(
            f"[PPO collect] rollout={self.rollout_index} "
            f"start target={self.rollout_target_steps} env_steps",
            flush=True,
        )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []
        num_envs = len(dones) if dones is not None else getattr(self.training_env, "num_envs", 1)
        if dones is not None:
            self.rollout_episodes += sum(bool(done) for done in dones)
        for info in infos:
            if "action_valid" in info:
                self.rollout_action_count += 1
                self.rollout_invalid_actions += int(not info["action_valid"])
            self.last_valid_action_count = info.get(
                "valid_action_count", self.last_valid_action_count
            )
            if info.get("termination_reason"):
                self.termination_reasons[info["termination_reason"]] += 1
        self.rollout_steps += num_envs
        if (
            self.rollout_steps % self.progress_interval == 0
            or self.rollout_steps >= self.rollout_target_steps
        ):
            invalid_rate = (
                self.rollout_invalid_actions / self.rollout_action_count
                if self.rollout_action_count
                else 0.0
            )
            print(
                f"[PPO collect] rollout={self.rollout_index} "
                f"collected={self.rollout_steps}/{self.rollout_target_steps} "
                f"total_timesteps={self.num_timesteps} "
                f"episodes_done={self.rollout_episodes} "
                f"invalid_rate={invalid_rate:.3f} "
                f"valid_actions={self.last_valid_action_count}",
                flush=True,
            )
        return True

    def _on_rollout_end(self) -> None:
        print(
            f"[PPO collect] rollout={self.rollout_index} complete; "
            f"terminations={dict(self.termination_reasons)}; PPO update starts",
            flush=True,
        )


class CheckpointCallback(BaseCallback):
    """Save resumable PPO checkpoints at a fixed timestep interval."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        save_freq: int,
        on_save=None,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.save_freq = max(1, int(save_freq))
        self.on_save = on_save
        self.last_saved_timesteps = 0

    def _save_checkpoint(self) -> None:
        if self.num_timesteps <= self.last_saved_timesteps:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.checkpoint_path.stem}.",
            suffix=self.checkpoint_path.suffix or ".zip",
            dir=self.checkpoint_path.parent,
        )
        os.close(temporary_fd)
        temporary_path = Path(temporary_name)
        try:
            self.model.save(str(temporary_path))
            saved_temporary_path = temporary_path
            if not saved_temporary_path.exists() and Path(
                f"{temporary_path}.zip"
            ).exists():
                saved_temporary_path = Path(f"{temporary_path}.zip")
            os.replace(saved_temporary_path, self.checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)
            Path(f"{temporary_path}.zip").unlink(missing_ok=True)
        self.last_saved_timesteps = int(self.num_timesteps)
        print(
            f"[PPO checkpoint] saved {self.checkpoint_path} "
            f"at {self.last_saved_timesteps} timesteps",
            flush=True,
        )
        if self.on_save is not None:
            self.on_save(self.checkpoint_path, self.last_saved_timesteps)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_saved_timesteps >= self.save_freq:
            self._save_checkpoint()
        return True

    def _on_training_end(self) -> None:
        self._save_checkpoint()


class LayoutAgent:
    """PPO agent for fixed-schema package layout environments."""

    def __init__(
        self,
        env,
        pre_trained_model_path: str = None,
        result_dir: str = None,
        *,
        design_env: QuNoCEnv = None,
        seed: int = None,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        batch_size: int = 64,
        ent_coef: float = 0.0,
        allow_legacy_model: bool = False,
        training_config: dict | None = None,
        output_model_path: str = None,
        run_manifest_path: str = None,
        task_domain: list[dict] | None = None,
        resource_provenance: dict | None = None,
    ):
        self.training_env = env
        self.env = design_env if design_env is not None else env
        self.result_dir = result_dir
        self.output_model_path = output_model_path
        self.run_manifest_path = run_manifest_path
        self.source_model_path = pre_trained_model_path
        self.task_domain = task_domain or []
        self.resource_provenance = resource_provenance or {}
        self.training_config = training_config or {}
        self.model = PPO(
            LayoutPolicy,
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            seed=seed,
        )
        if pre_trained_model_path:
            self.load_model(pre_trained_model_path, allow_legacy=allow_legacy_model)

    @property
    def trained_timesteps(self) -> int:
        return int(getattr(self.model, "num_timesteps", 0) or 0)

    def write_checkpoint_manifest(
        self,
        checkpoint_path: str | os.PathLike[str],
        trained_timesteps: int,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        manifest = build_model_manifest(
            self.env,
            trained_timesteps,
            self.source_model_path,
            self.training_config,
            output_model_path=str(checkpoint_path.resolve()),
            run_manifest_path=self.run_manifest_path,
            task_domain=self.task_domain,
            resource_provenance=self.resource_provenance,
        )
        manifest["artifact_role"] = "checkpoint"
        write_model_manifest(checkpoint_path, manifest)

    def train(
        self,
        total_timesteps: int = 10000,
        reset_num_timesteps: bool = True,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_on_save=None,
    ):
        callbacks = [RolloutProgressCallback()]
        if checkpoint_path is not None and checkpoint_interval is not None:
            callbacks.append(
                CheckpointCallback(
                    checkpoint_path,
                    checkpoint_interval,
                    on_save=checkpoint_on_save or self.write_checkpoint_manifest,
                )
            )
        callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        self.save_model()

    def train_until(
        self,
        target_timesteps: int,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_on_save=None,
    ):
        remaining_timesteps = int(target_timesteps) - self.trained_timesteps
        if remaining_timesteps <= 0:
            print(
                f"Skip training: model already has {self.trained_timesteps} "
                f"timesteps >= target {target_timesteps}."
            )
            self.save_model()
            return
        print(
            f"Continue training from {self.trained_timesteps} to {target_timesteps} timesteps."
        )
        self.train(
            remaining_timesteps,
            reset_num_timesteps=False,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            checkpoint_on_save=checkpoint_on_save,
        )

    def design(self, env: QuNoCEnv = None, seed: int = None) -> Package:
        design_env = env if env is not None else self.env
        print("Designing layout...")
        state, reset_info = design_env.reset(seed=seed)
        print(
            f"Initial score: {reset_info['score']}; "
            f"valid_actions: {reset_info['valid_action_count']}"
        )
        done = False
        while not done:
            action, _ = self.model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = design_env.step(action, record=True)
            done = terminated or truncated
            print(
                f"Action: raw={info['raw_action']} "
                f"semantic={info['semantic_action']} "
                f"valid={info['action_valid']} reward={reward:.6f} "
                f"score={info['score']:.6f} "
                f"valid_actions={info['valid_action_count']} "
                f"done={info['termination_reason']}",
                flush=True,
            )
        if design_env.best_feasible_package is None:
            raise NoFeasibleDesignError(
                "RL episode ended without satisfying the compute-ratio target."
            )
        return design_env.best_feasible_package

    def load_model(self, filepath: str = None, allow_legacy: bool = False):
        if filepath is None:
            model_path = os.path.join(self.result_dir, "model.zip")
            legacy_path = os.path.join(self.result_dir, "layout_agent.pth")
            if not os.path.exists(model_path) and os.path.exists(legacy_path):
                model_path = legacy_path
                allow_legacy = True
        else:
            model_path = filepath
        validate_model_manifest(model_path, self.env, allow_legacy=allow_legacy)
        self.model = PPO.load(model_path, env=self.training_env)

    def save_model(self, filepath: str = None):
        if filepath is not None:
            model_path = Path(filepath)
        elif self.output_model_path is not None:
            model_path = Path(self.output_model_path)
        elif self.result_dir is not None:
            model_path = Path(self.result_dir) / "model.zip"
        else:
            raise ValueError(
                "save_model requires filepath, output_model_path, or result_dir."
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = model_path.suffix or ".zip"
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{model_path.stem}.", suffix=suffix, dir=model_path.parent
        )
        os.close(temporary_fd)
        temporary_path = Path(temporary_name)
        try:
            self.model.save(str(temporary_path))
            saved_temporary_path = temporary_path
            if not saved_temporary_path.exists() and Path(
                f"{temporary_path}.zip"
            ).exists():
                saved_temporary_path = Path(f"{temporary_path}.zip")
            os.replace(saved_temporary_path, model_path)
        finally:
            temporary_path.unlink(missing_ok=True)
            Path(f"{temporary_path}.zip").unlink(missing_ok=True)
        manifest = build_model_manifest(
            self.env,
            self.trained_timesteps,
            self.source_model_path,
            self.training_config,
            output_model_path=str(model_path.resolve()),
            run_manifest_path=self.run_manifest_path,
            task_domain=self.task_domain,
            resource_provenance=self.resource_provenance,
        )
        write_model_manifest(model_path, manifest)
