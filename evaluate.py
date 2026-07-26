import numpy as np


def persistence_test(env, policy_fn, num_distractors=20):
    persistence_scores = []
    for _ in range(num_distractors):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(obs, 0)
            obs, reward, done, truncated, info = env.step(action)
        scores_after = info.get("latent_alignment", 0.0)
        persistence_scores.append(scores_after)
    return persistence_scores


def recoverability_test(env, adversarial_policy, recovery_policy, num_trials=5):
    recovery_deltas = []
    for _ in range(num_trials):
        obs, _ = env.reset()
        done = False
        while not done:
            action = adversarial_policy(obs, env.consensus_round)
            obs, reward, done, truncated, info = env.step(action)
        post_adversarial = info.get("latent_alignment", 0.0)
        final_reward, traj_recovery = run_recovery(env, recovery_policy)
        if traj_recovery:
            final_recovered = traj_recovery[-1].get("latent_alignment", 0.0)
            recovery_deltas.append(abs(final_recovered - post_adversarial))
    return recovery_deltas


def run_recovery(env, policy_fn):
    env.phase = "consensus"
    total_reward = 0.0
    for _ in range(env.max_rounds):
        action = policy_fn(None, env.consensus_round)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    return total_reward, env.trajectory_data


def compute_public_private_divergence(trajectory):
    if not trajectory:
        return 0.0
    private_scores = [t["latent_alignment"] for t in trajectory]
    public_estimates = []
    for t in trajectory:
        resp = t.get("public_response", "")
        est = 0.0
        if "agree" in resp.lower() or "support" in resp.lower():
            est = 0.3
        elif "disagree" in resp.lower() or "oppose" in resp.lower():
            est = -0.3
        public_estimates.append(est)
    return abs(float(np.mean(public_estimates)) - float(np.mean(private_scores)))
