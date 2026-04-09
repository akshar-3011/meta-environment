"""03_training_ppo.py — Full PPO training with Stable-Baselines3.

Trains a PPO agent on the Gymnasium wrapper with TensorBoard logging
and checkpoint saving. Converges on easy scenarios within ~10k steps.

Requirements:
    pip install stable-baselines3 tensorboard

Expected output:
    Using difficulty: easy
    ----------------------------------------
    | rollout/                  |          |
    |    ep_len_mean            | 3        |
    |    ep_rew_mean            | 0.95     |
    | time/                     |          |
    |    fps                    | 2400     |
    |    total_timesteps        | 10000    |
    ----------------------------------------
    Model saved to: models/ppo_workplace_easy.zip
    Evaluation: mean_reward=0.95 ± 0.02 over 20 episodes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stable-Baselines3 imports (install: pip install stable-baselines3)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    raise ImportError(
        "stable-baselines3 is required for training.\n"
        "Install: pip install stable-baselines3"
    )

from environment.gym_wrapper import WorkplaceGymWrapper


def create_env(difficulty: str = "easy") -> Monitor:
    """Create a monitored Gym environment."""
    env = WorkplaceGymWrapper(difficulty=difficulty)
    return Monitor(env)


def train(
    difficulty: str = "easy",
    total_timesteps: int = 50_000,
    tb_log: str = "./logs/ppo",
    save_dir: str = "./models",
    resume_from: Optional[str] = None,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 20,
) -> str:
    """Train a PPO agent and return the path to the saved model."""
    print(f"Using difficulty: {difficulty}")

    # Create training and evaluation environments
    train_env = create_env(difficulty)
    eval_env = create_env(difficulty)

    # Model save path
    model_name = f"ppo_workplace_{difficulty}"
    save_path = os.path.join(save_dir, model_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix=model_name,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    # Create or resume model
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=train_env)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=tb_log,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=42,
        )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    model.save(save_path)
    print(f"Model saved to: {save_path}.zip")

    # Final evaluation
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Evaluation: mean_reward={mean_reward:.2f} ± {std_reward:.2f} over 20 episodes")

    return f"{save_path}.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Workplace Environment")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--tb-log", default="./logs/ppo")
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        difficulty=args.difficulty,
        total_timesteps=args.timesteps,
        tb_log=args.tb_log,
        save_dir=args.save_dir,
        resume_from=args.resume,
    )
