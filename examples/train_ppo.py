#!/usr/bin/env python3
"""PPO training script for the Workplace Triage environment.

Usage:
    # Train on easy scenarios (quick convergence check):
    python examples/train_ppo.py --difficulty easy --timesteps 50000

    # Train on all scenarios:
    python examples/train_ppo.py --timesteps 100000

    # Train on hard scenarios with tensorboard:
    python examples/train_ppo.py --difficulty hard --timesteps 200000 --tb-log ./logs/

    # Resume training from checkpoint:
    python examples/train_ppo.py --resume models/ppo_workplace_easy/best_model.zip --timesteps 50000
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment.gym_wrapper import WorkplaceGymEnv


# ─── Custom Callbacks ───────────────────────────────────────────────────────

class RewardTracker(BaseCallback):
    """Track training metrics per episode and log summary statistics."""

    def __init__(self, log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.classify_rewards: List[float] = []
        self.reply_rewards: List[float] = []
        self.escalate_rewards: List[float] = []
        self.correct_classifications: int = 0
        self.correct_escalations: int = 0
        self.total_episodes: int = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(ep["r"])
                self.episode_lengths.append(ep["l"])
                self.total_episodes += 1

            # Track per-step metrics
            if "episode_rewards" in info and info.get("action_type") == "escalate":
                ep_rewards = info["episode_rewards"]
                self.classify_rewards.append(ep_rewards.get("classify", 0))
                self.reply_rewards.append(ep_rewards.get("reply", 0))
                self.escalate_rewards.append(ep_rewards.get("escalate", 0))

                # Check if classification was correct (> 0.35 means exact match)
                if ep_rewards.get("classify", 0) > 0.35:
                    self.correct_classifications += 1
                # Check if escalation was correct
                if ep_rewards.get("escalate", 0) > 0.2:
                    self.correct_escalations += 1

        if self.total_episodes > 0 and self.total_episodes % self.log_interval == 0:
            self._print_summary()

        return True

    def _print_summary(self):
        recent = self.episode_rewards[-self.log_interval:]
        c_recent = self.classify_rewards[-self.log_interval:]
        r_recent = self.reply_rewards[-self.log_interval:]
        e_recent = self.escalate_rewards[-self.log_interval:]

        mean_reward = np.mean(recent) if recent else 0
        classify_acc = (self.correct_classifications / max(self.total_episodes, 1)) * 100
        escalation_acc = (self.correct_escalations / max(self.total_episodes, 1)) * 100

        print(f"\n{'─' * 60}")
        print(f"  Episode {self.total_episodes} | Timestep {self.num_timesteps}")
        print(f"  Mean reward (last {self.log_interval}):  {mean_reward:.3f}")
        print(f"  Classify  mean: {np.mean(c_recent):.3f}" if c_recent else "")
        print(f"  Reply     mean: {np.mean(r_recent):.3f}" if r_recent else "")
        print(f"  Escalate  mean: {np.mean(e_recent):.3f}" if e_recent else "")
        print(f"  Classify accuracy:  {classify_acc:.1f}%")
        print(f"  Escalation accuracy: {escalation_acc:.1f}%")
        print(f"{'─' * 60}")


class ValidationCallback(BaseCallback):
    """Periodically validate against all difficulty tiers."""

    def __init__(self, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.validation_results: List[Dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            results = self._run_validation()
            self.validation_results.append(results)
            self._print_validation(results)
        return True

    def _run_validation(self) -> Dict:
        results = {}
        for diff in ["easy", "medium", "hard"]:
            env = WorkplaceGymEnv(difficulty_filter=diff)
            rewards = []
            classify_correct = 0
            escalate_correct = 0
            n_eval = 20

            for _ in range(n_eval):
                obs, _ = env.reset()
                total = 0.0
                ep_info = {}
                for _ in range(3):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    total += reward
                    ep_info = info
                rewards.append(total)
                if ep_info.get("episode_rewards", {}).get("classify", 0) > 0.35:
                    classify_correct += 1
                if ep_info.get("episode_rewards", {}).get("escalate", 0) > 0.2:
                    escalate_correct += 1

            results[diff] = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "classify_acc": classify_correct / n_eval * 100,
                "escalation_acc": escalate_correct / n_eval * 100,
            }
        results["timestep"] = self.num_timesteps
        return results

    def _print_validation(self, results: Dict):
        print(f"\n{'═' * 60}")
        print(f"  VALIDATION @ timestep {results['timestep']}")
        print(f"{'═' * 60}")
        for diff in ["easy", "medium", "hard"]:
            r = results[diff]
            print(f"  {diff:8s}  reward={r['mean_reward']:.3f}±{r['std_reward']:.3f}  "
                  f"classify={r['classify_acc']:.0f}%  escalation={r['escalation_acc']:.0f}%")

        # Check validation targets
        easy_r = results["easy"]["mean_reward"]
        hard_esc = results["hard"]["escalation_acc"]
        print()
        _check("Easy convergence (reward > 0.6)", easy_r > 0.6)
        _check("Hard escalation accuracy > 90%", hard_esc > 90)
        _check("Monotonicity (hard >= easy)", results["hard"]["mean_reward"] >= results["easy"]["mean_reward"] - 0.1)
        print(f"{'═' * 60}\n")


def _check(label: str, passed: bool):
    icon = "✅" if passed else "⏳"
    print(f"  {icon} {label}")


# ─── Training ───────────────────────────────────────────────────────────────

def make_env(difficulty: Optional[str] = None, seed: int = 0):
    def _init():
        env = WorkplaceGymEnv(difficulty_filter=difficulty)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(
    difficulty: Optional[str] = None,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    tb_log: Optional[str] = None,
    save_dir: str = "models",
    resume_path: Optional[str] = None,
    eval_freq: int = 5000,
    seed: int = 42,
):
    """Train PPO on the workplace triage environment."""

    diff_label = difficulty or "all"
    model_name = f"ppo_workplace_{diff_label}"
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Training PPO — difficulty={diff_label}, timesteps={total_timesteps:,}")
    print(f"   Save to: {save_path}")
    print(f"   Parallel envs: {n_envs}")
    if tb_log:
        print(f"   TensorBoard: {tb_log}")
    print()

    # Create vectorized environments
    env = DummyVecEnv([make_env(difficulty, seed=seed + i) for i in range(n_envs)])

    # Create eval environment
    eval_env = DummyVecEnv([make_env(difficulty, seed=seed + 100)])

    # Callbacks
    reward_tracker = RewardTracker(log_interval=200)
    validator = ValidationCallback(eval_freq=eval_freq)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(save_path / "eval_logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    callbacks = [reward_tracker, validator, eval_callback]

    # Create or load model
    if resume_path and Path(resume_path).exists():
        print(f"📂 Resuming from {resume_path}")
        model = PPO.load(resume_path, env=env, tensorboard_log=tb_log)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=tb_log,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=seed,
            policy_kwargs={
                "net_arch": dict(pi=[128, 128], vf=[128, 128]),
            },
        )

    # Train
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - start

    # Save final model
    final_path = save_path / "final_model"
    model.save(str(final_path))
    print(f"\n💾 Final model saved to {final_path}.zip")
    print(f"⏱  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Final validation
    print("\n" + "=" * 60)
    print("  FINAL VALIDATION")
    print("=" * 60)
    final_results = {}
    for diff in ["easy", "medium", "hard"]:
        test_env = WorkplaceGymEnv(difficulty_filter=diff)
        rewards = []
        classify_correct = 0
        escalate_correct = 0
        n_test = 50

        for _ in range(n_test):
            obs, _ = test_env.reset()
            total = 0.0
            ep_info = {}
            for _ in range(3):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = test_env.step(action)
                total += reward
                ep_info = info
            rewards.append(total)
            if ep_info.get("episode_rewards", {}).get("classify", 0) > 0.35:
                classify_correct += 1
            if ep_info.get("episode_rewards", {}).get("escalate", 0) > 0.2:
                escalate_correct += 1

        final_results[diff] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "classify_acc": classify_correct / n_test * 100,
            "escalation_acc": escalate_correct / n_test * 100,
        }
        r = final_results[diff]
        print(f"  {diff:8s}  reward={r['mean_reward']:.3f}±{r['std_reward']:.3f}  "
              f"[{r['min_reward']:.3f}, {r['max_reward']:.3f}]  "
              f"classify={r['classify_acc']:.0f}%  esc={r['escalation_acc']:.0f}%")

    print()
    easy_r = final_results["easy"]["mean_reward"]
    hard_esc = final_results["hard"]["escalation_acc"]
    _check(f"Easy convergence (reward={easy_r:.3f} > 0.6)", easy_r > 0.6)
    _check(f"Hard escalation accuracy ({hard_esc:.0f}% > 90%)", hard_esc > 90)
    mono = final_results["hard"]["mean_reward"] >= final_results["easy"]["mean_reward"] - 0.1
    _check(f"Reward monotonicity", mono)

    # Save results
    results_path = save_path / "training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "difficulty": diff_label,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "seed": seed,
            },
            "training_time_s": elapsed,
            "total_episodes": reward_tracker.total_episodes,
            "final_validation": final_results,
        }, f, indent=2)
    print(f"\n📊 Results saved to {results_path}")
    print("=" * 60)

    return model, final_results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO on Workplace Triage")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        default=None, help="Filter scenarios by difficulty (default: all)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100,000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--tb-log", type=str, default=None,
                        help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Model save directory (default: models/)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model checkpoint to resume from")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Validation frequency in timesteps (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        difficulty=args.difficulty,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        tb_log=args.tb_log,
        save_dir=args.save_dir,
        resume_path=args.resume,
        eval_freq=args.eval_freq,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
