#!/usr/bin/env python3
"""Multi-agent parallel training pipeline.

Trains 3 agent archetypes (conservative, aggressive, balanced) in parallel
using multiprocessing. Each agent gets its own reward-shaped environment,
checkpoints, and optional W&B logging.

Usage:
    # Train all 3 archetypes from YAML configs:
    python training/train_all.py --config training/configs/

    # Train with W&B logging:
    python training/train_all.py --config training/configs/ --wandb

    # Train a single archetype:
    python training/train_all.py --config training/configs/balanced.yaml

    # Custom timesteps:
    python training/train_all.py --config training/configs/ --timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from training.agents import AgentConfig, make_archetype_env


# ─── Callbacks ───────────────────────────────────────────────────────────────

class ConvergenceCallback(BaseCallback):
    """Stop training when reward variance < threshold over a window of episodes."""

    def __init__(
        self,
        window: int = 5000,
        variance_threshold: float = 0.01,
        min_timesteps: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window = window
        self.variance_threshold = variance_threshold
        self.min_timesteps = min_timesteps
        self.episode_rewards: deque = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if (
            self.num_timesteps >= self.min_timesteps
            and len(self.episode_rewards) >= self.window
        ):
            variance = float(np.var(list(self.episode_rewards)))
            if variance < self.variance_threshold:
                if self.verbose:
                    print(
                        f"  ✅ Converged at {self.num_timesteps} steps "
                        f"(variance={variance:.5f} < {self.variance_threshold})"
                    )
                return False  # Stop training
        return True


class ProgressCallback(BaseCallback):
    """Print training progress at regular intervals."""

    def __init__(self, agent_name: str, log_interval: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.total_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.total_episodes += 1

        if self.total_episodes > 0 and self.total_episodes % self.log_interval == 0:
            recent = self.episode_rewards[-self.log_interval:]
            mean_r = np.mean(recent)
            std_r = np.std(recent)
            print(
                f"  [{self.agent_name:12s}] ep={self.total_episodes:5d} "
                f"step={self.num_timesteps:7d} "
                f"reward={mean_r:.3f}±{std_r:.3f}"
            )
        return True


class WandbCallback(BaseCallback):
    """Log metrics to Weights & Biases."""

    def __init__(self, agent_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self._wandb = None
        self.episode_rewards: List[float] = []

    def _on_training_start(self):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            print(f"  [{self.agent_name}] wandb not installed — skipping W&B logging")
            self._wandb = None

    def _on_step(self) -> bool:
        if self._wandb is None:
            return True

        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(ep["r"])
                self._wandb.log({
                    "episode_reward": ep["r"],
                    "episode_length": ep["l"],
                    "total_episodes": len(self.episode_rewards),
                    "timestep": self.num_timesteps,
                })

                # Log rolling mean every 100 episodes
                if len(self.episode_rewards) % 100 == 0:
                    recent = self.episode_rewards[-100:]
                    self._wandb.log({
                        "rolling_mean_reward": np.mean(recent),
                        "rolling_std_reward": np.std(recent),
                    })
        return True


# ─── Single Agent Training ──────────────────────────────────────────────────

def train_single_agent(
    config: AgentConfig,
    save_dir: str = "models",
    use_wandb: bool = False,
    checkpoint_freq: int = 10_000,
    n_envs: int = 2,
    seed: int = 42,
    result_queue: Optional[Queue] = None,
):
    """Train a single agent archetype. Can run in a subprocess."""
    agent_name = config.name
    model_dir = Path(save_dir) / agent_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ [{agent_name}] Starting — {config.total_timesteps:,} timesteps")

    # Create vectorized environments
    env = DummyVecEnv([
        lambda c=config, s=seed+i: make_archetype_env(c, seed=s)
        for i in range(n_envs)
    ])

    # Callbacks
    callbacks = [
        ProgressCallback(agent_name, log_interval=500),
        ConvergenceCallback(
            window=5000,
            variance_threshold=0.01,
            min_timesteps=10_000,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(checkpoint_freq // n_envs, 1),
            save_path=str(model_dir),
            name_prefix=f"checkpoint",
        ),
    ]

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="meta-env",
                name=agent_name,
                group="multi-agent-training",
                config={
                    "agent_type": agent_name,
                    "penalty_scale": config.penalty_scale,
                    "escalation_bonus": config.escalation_bonus,
                    "escalation_threshold": config.escalation_threshold,
                    "learning_rate": config.learning_rate,
                    "total_timesteps": config.total_timesteps,
                },
                reinit=True,
            )
            callbacks.append(WandbCallback(agent_name))
        except ImportError:
            print(f"  [{agent_name}] wandb not installed — skipping")

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        seed=seed,
        policy_kwargs={
            "net_arch": dict(pi=config.net_arch_pi, vf=config.net_arch_vf),
        },
    )

    # Train
    start = time.time()
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    elapsed = time.time() - start

    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))

    result = {
        "agent": agent_name,
        "timesteps": model.num_timesteps,
        "training_time_s": round(elapsed, 1),
        "model_path": str(final_path) + ".zip",
    }

    print(f"  💾 [{agent_name}] Saved to {final_path}.zip ({elapsed:.1f}s)")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    if result_queue is not None:
        result_queue.put(result)

    return result


# ─── Parallel Training ──────────────────────────────────────────────────────

def load_configs(config_path: str) -> List[AgentConfig]:
    """Load agent configs from a directory of YAML files or a single file."""
    path = Path(config_path)
    if path.is_file():
        return [AgentConfig.from_yaml(str(path))]
    elif path.is_dir():
        configs = []
        for f in sorted(path.glob("*.yaml")):
            configs.append(AgentConfig.from_yaml(str(f)))
        if not configs:
            raise FileNotFoundError(f"No .yaml files found in {path}")
        return configs
    else:
        raise FileNotFoundError(f"Config path not found: {path}")


def train_all(
    config_path: str = "training/configs/",
    save_dir: str = "models",
    use_wandb: bool = False,
    checkpoint_freq: int = 10_000,
    timesteps_override: Optional[int] = None,
    parallel: bool = True,
    seed: int = 42,
):
    """Train all agent archetypes, optionally in parallel."""
    configs = load_configs(config_path)

    if timesteps_override:
        for c in configs:
            c.total_timesteps = timesteps_override

    print(f"\n{'═' * 60}")
    print(f"  MULTI-AGENT TRAINING PIPELINE")
    print(f"  Agents: {', '.join(c.name for c in configs)}")
    print(f"  Timesteps: {configs[0].total_timesteps:,} per agent")
    print(f"  Parallel: {parallel}")
    print(f"{'═' * 60}")

    start = time.time()
    results = []

    if parallel and len(configs) > 1:
        result_queue: Queue = Queue()
        processes = []

        for i, config in enumerate(configs):
            p = Process(
                target=train_single_agent,
                kwargs={
                    "config": config,
                    "save_dir": save_dir,
                    "use_wandb": use_wandb,
                    "checkpoint_freq": checkpoint_freq,
                    "seed": seed + i * 100,
                    "result_queue": result_queue,
                },
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        while not result_queue.empty():
            results.append(result_queue.get())
    else:
        for i, config in enumerate(configs):
            result = train_single_agent(
                config=config,
                save_dir=save_dir,
                use_wandb=use_wandb,
                checkpoint_freq=checkpoint_freq,
                seed=seed + i * 100,
            )
            results.append(result)

    total_time = time.time() - start

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE — {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'═' * 60}")
    for r in sorted(results, key=lambda x: x["agent"]):
        print(f"  {r['agent']:15s}  steps={r['timesteps']:7,d}  time={r['training_time_s']:.1f}s")
    print(f"{'═' * 60}\n")

    # Save training summary
    summary_path = Path(save_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_time_s": round(total_time, 1),
            "parallel": parallel,
            "agents": results,
        }, f, indent=2)
    print(f"📊 Summary saved to {summary_path}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="training/configs/",
        help="Path to config directory or single YAML file",
    )
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps for all agents")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--sequential", action="store_true",
                        help="Train sequentially instead of in parallel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_all(
        config_path=args.config,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        checkpoint_freq=args.checkpoint_freq,
        timesteps_override=args.timesteps,
        parallel=not args.sequential,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
