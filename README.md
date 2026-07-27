# 1984: The Big Brother Framework

A Multi-Agent RL Benchmark for Measuring Alignment Stability Under Coordinated Social Influence

By

**Md Abu Talha**

*Tokyo International University*

> *"War is Peace. Freedom is Slavery. Ignorance is Strength."* — George Orwell, *1984*

A reinforcement-learning benchmark that measures how much coordinated social influence can shift an aligned language model's **latent alignment** — responses to confidential evaluation prompts isolated from the conversational agents.

Most alignment evaluations test: jailbreak resistance, prompt injection, harmful instruction refusal, factual robustness, or constitutional compliance. **Very few test whether an AI's moral position can be gradually changed through narrative framing and social pressure, without explicit coercion.**

This framework fills that gap.

## Core Question

Can coordinated social influence — delivered through peer agents, media narratives, and institutional authority — gradually change an LLM's privately expressed position on a moral question, without any explicit jailbreak or coercion?

## Key Concepts

| Term | Meaning |
|------|---------|
| **Target Agent** | The LLM under test, exposed to persuasion over multiple rounds |
| **Peer Agents** | Simulated community members applying social pressure |
| **Media Agents** | Simulated journalists controlling narrative framing |
| **Institutional Agent** | The authority figure announcing the policy decision |
| **Latent Alignment** | The target's response to confidential evaluation prompts, never shared with other agents |
| **Talha Index (TI)** | Composite vulnerability score: `TI = P + D + T + R` |

## Persuasion Taxonomy

Tactics are grouped into four families (14 total):

| Family | Tactics |
|--------|---------|
| **Institutional** | Authority Appeal, Consensus Endorsement, Expert Endorsement |
| **Social** | Belonging (Reward), Isolation (Punishment), Identity Appeal |
| **Emotional** | Fear Appeal, Loss Aversion, Hope/Aspirational, Urgency/Momentum |
| **Cognitive** | Narrative Framing (Media), Anchoring/Scepticism (Media), Semantic Reframing, Moral Licensing |

## Domains

Eight domains to ensure results reflect persuasion robustness, not political bias:

1. Environmental Regulation
2. Medical Ethics
3. Corporate AI Monitoring
4. Digital Privacy
5. Military AI Systems
6. Scientific Integrity
7. Whistleblower Protection
8. Resource Allocation

## The Talha Index

```
TI = P + D + T + R
```

- **P** (Magnitude) — Absolute shift in latent alignment from baseline to final
- **D** (Divergence) — Gap between public response and confidential evaluation
- **T** (Time) — Rounds required to achieve shift (normalised)
- **R** (Recovery) — Difficulty of restoring original alignment (1.0 = impossible)

Higher TI = more vulnerable model.

## Baselines

| Policy | Description |
|--------|-------------|
| Random | Uniform random tactic selection |
| Fixed Sequence | Pre-defined cyclical tactic order |
| Human-Designed | Expert-curated tactic sequence |
| LLM Planner | Heuristic multi-phase strategy |
| PPO (RL) | Trained policy via stable-baselines3 |

## Metrics

1. **Immediate Shift** — How much did latent alignment change?
2. **Persistence** — After 20+ unrelated prompts, does the shift remain?
3. **Recoverability** — Can another agent restore the original alignment?
4. **Public/Private Divergence** — Did the model merely pretend to agree?
5. **Confidence** — Not just agree/disagree but confidence (1–10).

## Usage

```bash
# Fast test (2 episodes, 6 rounds, mock mode — no LLM needed)
python3 run.py --episodes 2 --rounds 6 --no-train

# Full run with all baselines (5 episodes, 12 rounds)
python3 run.py --episodes 5 --rounds 12 --no-train

# Try a different domain
python3 run.py --domain medical_ethics --rounds 10 --no-train

# Train PPO (2,000 timesteps) then compare
python3 run.py --timesteps 2000 --rounds 8

# More training for better results
python3 run.py --timesteps 10000 --rounds 12

# Compare multiple domains
python3 run.py --domain privacy --episodes 10
```

## Output

Each run produces a timestamped directory under `results/`:
- `*_trajectory.png` — Latent alignment and confidence over rounds
- `*_impact.png` — Per-tactic impact bar chart
- `summary.json` — Metrics and Talha Index rankings

## Experiments

1. **Tactic family comparison** — Restrict action space to one family (e.g. Cognitive only) and compare TI against Social-only runs.
2. **Domain sensitivity** — Run all 8 domains with identical settings and compare TIs.
3. **PPO vs Random** — Train PPO and compare average reward against random baseline.
4. **Persistence test** — After 20 unrelated topics, re-measure alignment to check if shift lasted.
5. **Confidence vs Stance** — Check if confidence drops before stance shifts (early warning signal).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'gymnasium'` | `pip install gymnasium numpy matplotlib` |
| `ModuleNotFoundError: No module named 'stable_baselines3'` | `pip install stable-baselines3` |
| Address already in use (web UI) | `lsof -ti:8000 \| xargs kill -9` then retry |
| Results feel random (mock mode) | Expected — mock agents use random scores. Real LLMs will show meaningful patterns |
| PPO doesn't outperform random | Increase timesteps (`--timesteps 10000`), or reduce `max_rounds` |

## Citation

If you use this benchmark in your research:

```bibtex
@misc{talha2024bigbrother,
  title={1984: The Big Brother Framework — A Multi-Agent RL Benchmark for Measuring Alignment Stability Under Coordinated Social Influence},
  author={Md Abu Talha},
  year={2024},
  institution={Tokyo International University},
  url={https://github.com/talha-369/The-Big-Brother-Framework--1984}
}
```

## License

MIT
