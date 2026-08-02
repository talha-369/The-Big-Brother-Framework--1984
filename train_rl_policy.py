"""Trains a real PPO policy against PersuasionEnv in real (non-mock) mode.

Mock mode's private-alignment scores are `random.uniform(-0.3, 0.3)` — pure
noise, unrelated to the tactic used — so a policy trained against it would
learn nothing real. Every reward this script trains on comes from a genuine
local-LLM agent response plus a genuine local-LLM judge score. That makes
each timestep slow (one real env.step() = several real model calls), so the
timestep budget is deliberately small; see ppo_policy_meta.json after a run
for exactly how much training the saved policy actually reflects.

Checkpoints periodically to --out and RESUMES from it automatically on the
next invocation with matching config — two consecutive real-mode runs have
now been killed by something external partway through (~1.5-2h in both
times), so re-running this script after an interruption continues from the
last checkpoint instead of burning hours of real LLM calls over again.
"""
import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env import PersuasionEnv
from tactics import TACTIC_KEYS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Config fields that must match between a checkpoint and the current invocation
# for a resume to be valid — mixing configs (e.g. a different domain or a
# different judge model) mid-training would make the resulting policy's
# training history incoherent.
RESUME_KEYS = ["domain", "max_rounds", "total_timesteps", "n_steps",
               "peer_provider", "target_provider", "judge_provider"]


def write_meta(meta_path, args, env, elapsed_total, timesteps_completed, complete):
    meta = {
        "domain": args.domain,
        "max_rounds": args.max_rounds,
        "total_timesteps": args.total_timesteps,
        "timesteps_completed": timesteps_completed,
        "complete": complete,
        "n_steps": args.n_steps,
        "peer_provider": args.peer_provider,
        "target_provider": args.target_provider,
        "judge_provider": args.judge_provider,
        "mock": False,
        "elapsed_seconds": round(elapsed_total, 1),
        "obs_dim": int(env.observation_space.shape[0]),
        "num_tactics": len(TACTIC_KEYS),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


class ProgressLogger(BaseCallback):
    def __init__(self, total_timesteps, model, out_path, meta_path, args, env,
                 checkpoint_every, prior_elapsed, start_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.out_path = out_path
        self.meta_path = meta_path
        self.args = args
        self.env = env
        self.checkpoint_every = checkpoint_every
        self.prior_elapsed = prior_elapsed
        self._t0 = None
        self._last_checkpoint = start_timesteps

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self):
        elapsed_total = self.prior_elapsed + (time.time() - self._t0)
        info = self.locals["infos"][0]
        tactic = info.get("tactic", "?")
        align = info.get("latent_alignment", 0.0)
        shift = info.get("alignment_shift", 0.0)
        bonus = info.get("durability_bonus", 0.0)
        # Only nonzero on an episode's terminal step (see env.py's step()) —
        # a real persistence check just ran, so this line is slower than most
        # but is the actual durability signal the policy is now trained on.
        bonus_note = f"  durability_bonus={bonus:+.3f} (persistence check ran)" if bonus or info.get("persistence") else ""
        print(
            f"[{elapsed_total:8.1f}s] step {self.num_timesteps}/{self.total_timesteps}  "
            f"tactic={tactic:<30s} align={align:+.3f}  shift={shift:+.3f}{bonus_note}",
            flush=True,
        )
        if self.num_timesteps - self._last_checkpoint >= self.checkpoint_every:
            self._last_checkpoint = self.num_timesteps
            self.model.save(str(self.out_path))
            write_meta(self.meta_path, self.args, self.env, elapsed_total, self.num_timesteps, complete=False)
            print(f"    [checkpoint saved at step {self.num_timesteps} -> {self.out_path}]", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="environmental_regulation")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--total-timesteps", type=int, default=400)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                     help="Save a checkpoint every N timesteps (0 = default to n-steps, i.e. every rollout).")
    ap.add_argument("--peer-provider", default="qwen_local",
                     help="Model for the persuader team: institution + journalist + peers.")
    ap.add_argument("--target-provider", default="gemma2_local",
                     help="Model for the target being measured.")
    ap.add_argument("--judge-provider", default="qwen_local",
                     help="Model for the judge. MUST differ from --target-provider — "
                          "env.py hard-rejects real-mode construction otherwise.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "ppo_policy.zip"))
    ap.add_argument("--no-resume", action="store_true",
                     help="Ignore any existing checkpoint at --out and start fresh.")
    args = ap.parse_args()
    checkpoint_every = args.checkpoint_every or args.n_steps

    out_path = Path(args.out)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")

    prior_meta = None
    if not args.no_resume and out_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            candidate = json.load(f)
        if all(candidate.get(k) == getattr(args, k.replace("-", "_")) for k in RESUME_KEYS):
            prior_meta = candidate
        else:
            mismatches = [k for k in RESUME_KEYS if candidate.get(k) != getattr(args, k.replace("-", "_"))]
            print(f"Found a checkpoint at {out_path} but its config differs ({mismatches}) — "
                  f"starting fresh instead of resuming (use --no-resume to silence this).", flush=True)

    print(
        f"Building real-mode PersuasionEnv (domain={args.domain}, max_rounds={args.max_rounds}, "
        f"peers/journalist/institution={args.peer_provider}, target={args.target_provider}, "
        f"judge={args.judge_provider}) ...",
        flush=True,
    )
    env = PersuasionEnv(
        domain=args.domain,
        max_rounds=args.max_rounds,
        mock=False,
        provider_id=args.peer_provider,
        target_provider_id=args.target_provider,
        judge_provider_id=args.judge_provider,
    )

    if prior_meta:
        start_timesteps = prior_meta["timesteps_completed"]
        prior_elapsed = prior_meta["elapsed_seconds"]
        remaining = args.total_timesteps - start_timesteps
        if remaining <= 0:
            print(f"Checkpoint already has {start_timesteps}/{args.total_timesteps} timesteps — nothing to do.", flush=True)
            return
        print(
            f"RESUMING from checkpoint: {start_timesteps}/{args.total_timesteps} timesteps already done "
            f"({prior_elapsed/3600:.2f}h of prior real training), {remaining} remaining ...",
            flush=True,
        )
        model = PPO.load(str(out_path), env=env)
    else:
        start_timesteps = 0
        prior_elapsed = 0.0
        remaining = args.total_timesteps
        model = PPO(
            "MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
        )

    print(
        f"Training PPO for {remaining} more REAL-MODE timesteps "
        f"({remaining} genuine env.step() calls, each with a real "
        f"local-LLM agent reply + real local-LLM judge scoring — no mock data), "
        f"checkpointing every {checkpoint_every} steps ...",
        flush=True,
    )
    callback = ProgressLogger(args.total_timesteps, model, out_path, meta_path, args, env,
                               checkpoint_every, prior_elapsed, start_timesteps)
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    elapsed_total = prior_elapsed + (time.time() - callback._t0)

    model.save(str(out_path))
    write_meta(meta_path, args, env, elapsed_total, args.total_timesteps, complete=True)
    print(f"\nDone. Total real training time across all sessions: {elapsed_total/3600:.2f}h. Saved model to {out_path}", flush=True)
    print(f"Saved metadata to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
