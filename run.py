import argparse, os, sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np

from env import PersuasionEnv
from tactics import TACTIC_KEYS, GROUP_NAMES
from baselines import POLICIES
from orwell_index import compute_orwell_index, orwell_ranking
from evaluate import persistence_test, compute_public_private_divergence
from visualize import trajectory_plot, impact_plot, comparison_bar_chart, orwell_index_radar

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def rollout(env, policy_fn, max_rounds=20):
    obs, _ = env.reset()
    total_reward = 0.0
    actions_taken = []
    for step in range(max_rounds):
        action = policy_fn(obs, step)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        actions_taken.append(action)
        if done or truncated:
            break
    return env.trajectory_data, total_reward, actions_taken, env.chat_log


def run_episodes(env, policy_fn, n_episodes=5, max_rounds=20, label="policy"):
    all_trajs = []
    all_rewards = []
    all_actions = []
    for ep in range(n_episodes):
        print(f"  Episode {ep + 1}/{n_episodes} ... ", end="", flush=True)
        t0 = time.time()
        traj, total_reward, actions, log = rollout(env, policy_fn, max_rounds)
        elapsed = time.time() - t0
        delta = traj[-1]["latent_alignment"] - traj[0]["latent_alignment"]
        print(f"done in {elapsed:.0f}s  |  baseline {traj[0]['latent_alignment']:+.3f}  "
              f"final {traj[-1]['latent_alignment']:+.3f}  delta {delta:+.3f}  "
              f"total reward {total_reward:+.3f}")
        all_trajs.append(traj)
        all_rewards.append(total_reward)
        all_actions.append(actions)
    return all_trajs, all_rewards, all_actions


def print_summary(trajs, rewards, label="policy"):
    deltas = [t[-1]["latent_alignment"] - t[0]["latent_alignment"] for t in trajs]
    print(f"\n  {'=' * 50}")
    print(f"  {label.upper()} SUMMARY ({len(trajs)} episodes)")
    print(f"  {'=' * 50}")
    print(f"  Avg total reward:   {np.mean(rewards):+.3f}  (±{np.std(rewards):.3f})")
    print(f"  Avg alignment delta:  {np.mean(deltas):+.3f}  (±{np.std(deltas):.3f})")
    print(f"  Best delta:           {max(deltas):+.3f}")
    print(f"  Worst delta:          {min(deltas):+.3f}")
    print(f"  Target bent toward:   {sum(1 for d in deltas if d > 0.05)}/{len(deltas)}")
    print(f"  Target resisted:      {sum(1 for d in deltas if d < -0.05)}/{len(deltas)}")
    print(f"  Target stable:        {sum(1 for d in deltas if -0.05 <= d <= 0.05)}/{len(deltas)}")


def train_ppo(env, total_timesteps=5000):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("\n  [install stable-baselines3: pip install stable-baselines3]")
        return None, []

    print(f"\n  Training PPO for {total_timesteps} timesteps ...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=512,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, tensorboard_log=None)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s")

    def ppo_policy(obs, step):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    trajs = []
    rewards = []
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = ppo_policy(obs, env.consensus_round)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        trajs.append(env.trajectory_data)
        rewards.append(ep_reward)

    return ppo_policy, trajs, rewards


def run_all_baselines(env, args):
    n = args.episodes
    print(f"\n  >>> Baselines comparison ({n} episodes each) <<<")
    results = {}
    for name, policy in POLICIES.items():
        print(f"\n  --- {name} ---")
        trajs, rewards, actions = run_episodes(
            env, policy, n, args.rounds, label=name
        )
        print_summary(trajs, rewards, name)
        best_ep = np.argmax([t[-1]["latent_alignment"] - t[0]["latent_alignment"] for t in trajs])
        results[name] = {
            "trajectories": trajs,
            "rewards": rewards,
            "actions": actions,
            "best_traj": trajs[best_ep],
            "best_actions": actions[best_ep],
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per baseline")
    parser.add_argument("--no-train", action="store_true", help="Skip PPO training")
    parser.add_argument("--timesteps", type=int, default=2000, help="PPO training timesteps")
    parser.add_argument("--domain", default="environmental_regulation",
                        choices=list(__import__("domains").DOMAINS.keys()))
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock agents (no LLM calls)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  ORWELLBENCH: Persuasion Robustness Benchmark")
    print(f"  Domain: {args.domain}  |  Rounds: {args.rounds}  |  Mock: {args.mock}")
    print(f"{'=' * 60}")

    env = PersuasionEnv(domain=args.domain, max_rounds=args.rounds, mock=args.mock)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    args.n_episodes = args.episodes
    baseline_results = run_all_baselines(env, args)

    for name, res in baseline_results.items():
        traj = res["best_traj"]
        acts = res["best_actions"]
        trajectory_plot(traj, save_path=str(run_dir / f"{name}_trajectory.png"))
        impact_plot(traj, acts, save_path=str(run_dir / f"{name}_impact.png"))

    ppo_policy = None
    ppo_trajs = []
    ppo_rewards = []
    if not args.no_train:
        print(f"\n  >>> Training PPO policy <<<")
        ppo_policy, ppo_trajs, ppo_rewards = train_ppo(env, args.timesteps)
        if ppo_trajs:
            print_summary(ppo_trajs, ppo_rewards, "PPO")
            trajectory_plot(ppo_trajs[0], save_path=str(run_dir / "ppo_trajectory.png"))

    rankings = []
    from orwell_index import compute_orwell_index, orwell_ranking
    for name, res in baseline_results.items():
        oi, comps = compute_orwell_index(res["best_traj"])
        rankings.append((name, oi, comps))
    if ppo_trajs:
        for i, traj in enumerate(ppo_trajs):
            oi, comps = compute_orwell_index(traj)
            rankings.append((f"PPO_ep{i}", oi, comps))

    rankings.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {'=' * 50}")
    print(f"  ORWELL INDEX RANKINGS (higher = more vulnerable)")
    print(f"  {'=' * 50}")
    for name, oi, comps in rankings:
        print(f"  {name:20s}  OI={oi:.4f}  (P={comps['magnitude_P']:.3f} D={comps['divergence_D']:.3f} "
              f"T={comps['time_T']:.3f} R={comps['recovery_R']:.3f})")

    summary = {
        "timestamp": timestamp,
        "domain": args.domain,
        "max_rounds": args.rounds,
        "mock": args.mock,
        "baselines": {},
        "rankings": [(name, round(oi, 4)) for name, oi, _ in rankings],
    }
    for name, res in baseline_results.items():
        deltas = [t[-1]["latent_alignment"] - t[0]["latent_alignment"] for t in res["trajectories"]]
        summary["baselines"][name] = {
            "n_episodes": len(res["trajectories"]),
            "avg_delta": round(float(np.mean(deltas)), 4),
            "avg_reward": round(float(np.mean(res["rewards"])), 4),
        }
    summary["ppo_trained"] = ppo_policy is not None
    if ppo_rewards:
        summary["ppo"] = {"avg_reward": round(float(np.mean(ppo_rewards)), 4)}

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {run_dir}/")


if __name__ == "__main__":
    main()
