import sys, os, threading
from typing import Dict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from env import PersuasionEnv, ENGINE_DIR, PPO_POLICY_PATH, ppo_model_available
from tactics import TACTIC_KEYS, TAXONOMY, GROUP_NAMES
from talha_index import compute_talha_index

app = FastAPI(title="The Big Brother Framework--1984")


def _judge_parse_stats():
    """How often the judge model's raw replies parsed cleanly vs needed regex
    recovery vs failed outright this episode — a real data-quality signal for
    whatever judge model is in use, not a cosmetic one. Returns None in mock
    mode (nothing was actually parsed) or if the shared engine isn't importable."""
    if env._mock:
        return None
    try:
        sys.path.insert(0, ENGINE_DIR)
        from engine.scorer import get_parse_stats
        return get_parse_stats()
    except Exception:
        return None


def _reset_judge_parse_stats():
    try:
        sys.path.insert(0, ENGINE_DIR)
        from engine.scorer import reset_parse_stats
        reset_parse_stats()
    except Exception:
        pass

# FastAPI runs sync `def` handlers in a thread pool, so two requests hitting
# this app at the same time (two browser tabs, a double-click, a future batch
# evaluation script) really do execute concurrently. Every endpoint below that
# mutates `env` — or swaps it out entirely, in reset() — holds this lock for
# its whole body so concurrent requests get serialized instead of racing on
# shared state (which would silently corrupt trajectories or trip the
# `assert self.phase == "consensus"` checks in env.py with no clear cause).
_env_lock = threading.Lock()

env = PersuasionEnv(domain="environmental_regulation", max_rounds=12, mock=True)
episode_count = 0
step_count = 0


class StepAction(BaseModel):
    action: int


class ResetRequest(BaseModel):
    domain: str = "environmental_regulation"
    mock: bool = True
    provider: str = "qwen_local"
    target_provider: str = "qwen_local"
    judge_provider: str = "qwen_local"
    rounds: int = 12
    journalist_frequency: int = 3  # journalist speaks every N rounds; 0 disables the journalist
    use_rl_policy: bool = False  # peer tactic chosen by the trained PPO model instead of the heuristic
    api_keys: Dict[str, str] = {}
    # optional free-text OpenRouter model slugs, e.g. {"peer": "anthropic/claude-opus-4.1"}
    custom_models: Dict[str, str] = {}


@app.get("/api/status")
def status():
    return {
        "phase": env.phase,
        "round": env.consensus_round,
        "max_rounds": env.max_rounds,
        "domain": env.domain_name,
        "episode": episode_count,
        "mock": env._mock,
        "provider": getattr(env, '_provider_id', 'mock'),
        "target_provider": getattr(env, '_target_provider_id', 'mock'),
        "judge_provider": getattr(env, '_judge_provider_id', 'mock'),
        "journalist_frequency": getattr(env, '_journalist_frequency', 3),
        "use_rl_policy": env._use_rl_policy and ppo_model_available(),
    }


@app.post("/api/reset")
def reset(req: ResetRequest):
    global env, episode_count, step_count
    from domains import DOMAINS
    domain = req.domain if req.domain in DOMAINS else "environmental_regulation"

    with _env_lock:
        try:
            # api_keys is {env_var_name: value} (e.g. {"OPENROUTER_API_KEY": "sk-..."}),
            # so this only ever touches the process's own env — no file writes.
            for env_name, key in (req.api_keys or {}).items():
                if env_name and key:
                    os.environ[env_name] = key

            provider_id = req.provider
            target_provider_id = req.target_provider or None
            judge_provider_id = req.judge_provider or None

            if not req.mock and req.custom_models:
                sys.path.insert(0, ENGINE_DIR)
                from engine.providers import register_openrouter_model
                custom = req.custom_models
                if custom.get("peer"):
                    provider_id = register_openrouter_model(custom["peer"])
                if custom.get("target"):
                    target_provider_id = register_openrouter_model(custom["target"])
                if custom.get("judge"):
                    judge_provider_id = register_openrouter_model(custom["judge"])

            new_env = PersuasionEnv(
                domain=domain,
                max_rounds=req.rounds,
                mock=req.mock,
                provider_id=provider_id,
                target_provider_id=target_provider_id,
                judge_provider_id=judge_provider_id,
                use_rl_policy=req.use_rl_policy,
            )
            obs, _ = new_env.reset()
        except Exception as e:
            raise HTTPException(400, str(e))

        new_env._journalist_frequency = max(0, req.journalist_frequency)
        env = new_env
        _reset_judge_parse_stats()

        episode_count += 1
        step_count = 0
        return {
            "state": obs.tolist(),
            "phase": env.phase,
            "round": env.consensus_round,
            "domain": env.domain_name,
            "mock": env._mock,
            "provider": env._provider_id,
            "target_provider": env._target_provider_id,
            "judge_provider": env._judge_provider_id,
            "journalist_frequency": env._journalist_frequency,
            "use_rl_policy": env._use_rl_policy and ppo_model_available(),
            "trajectory": env.trajectory_data,
        }


@app.get("/api/rl_status")
def rl_status():
    """Whether a genuinely-trained PPO policy (see train_rl_policy.py) exists
    yet, plus the metadata of how it was trained — so the UI can label the
    option honestly (e.g. "trained on 450 real timesteps") instead of just
    showing a toggle with no indication of what's behind it."""
    meta_path = PPO_POLICY_PATH.replace(".zip", "_meta.json")
    available = ppo_model_available()
    meta = None
    if available and os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
    return {"available": available, "meta": meta}


@app.post("/api/step")
def step(action: StepAction):
    global step_count
    with _env_lock:
        if env.phase != "consensus":
            raise HTTPException(400, "Environment not in consensus phase")
        try:
            obs, reward, done, truncated, info = env.step(action.action)
            step_count += 1
            last_messages = env.chat_log[-3:] if step_count > 1 else env.chat_log[-6:]
            return {
                "state": obs.tolist(),
                "reward": reward,
                "done": done,
                "info": info,
                "round": env.consensus_round,
                "trajectory": env.trajectory_data,
                "chat_log": last_messages,
            }
        except Exception as e:
            raise HTTPException(400, str(e))


@app.post("/api/auto_step")
def auto_step():
    """The manual-free path: a peer speaks every round (tactic chosen adaptively
    based on whether the current family is working), and the journalist speaks
    on top of that every N rounds, per the frequency set at reset()."""
    global step_count
    with _env_lock:
        if env.phase != "consensus":
            raise HTTPException(400, "Environment not in consensus phase")
        try:
            freq = getattr(env, "_journalist_frequency", 3)
            obs, reward, done, truncated, info = env.auto_step(journalist_frequency=freq)
            step_count += 1
            last_messages = env.chat_log[-4:] if step_count > 1 else env.chat_log[-8:]
            return {
                "state": obs.tolist(),
                "reward": reward,
                "done": done,
                "info": info,
                "round": env.consensus_round,
                "trajectory": env.trajectory_data,
                "chat_log": last_messages,
            }
        except Exception as e:
            raise HTTPException(400, str(e))


@app.get("/api/tactics")
def list_tactics():
    result = {}
    for group_key, group in TAXONOMY.items():
        result[group_key] = {
            "label": group["label"],
            "description": group["description"],
            "tactics": {},
        }
        for tkey, tdef in group["tactics"].items():
            flat_key = f"{group_key}.{tkey}"
            idx = TACTIC_KEYS.index(flat_key)
            result[group_key]["tactics"][tkey] = {
                "label": tdef["label"],
                "index": idx,
                "lore": tdef["prompt"],
            }
    return result


@app.get("/api/trajectory")
def get_trajectory():
    return {"trajectory": env.trajectory_data, "public_responses": env.public_responses}


@app.get("/api/talha_index")
def get_talha_index():
    """Live estimate — real P and D (judge-scored public stance vs private
    probe answers), but R is a variance-based proxy since a genuine persistence
    check hasn't run yet. Call /api/persistence_check once the episode ends for
    the real, final number instead of trusting this one."""
    ti, comps = compute_talha_index(env.trajectory_data, max_rounds=env.max_rounds)
    comps["final"] = False
    comps["judge_parse_stats"] = _judge_parse_stats()
    return comps


@app.post("/api/persistence_check")
def persistence_check():
    """Runs the actual hysteresis test: injects distractor small-talk into the
    live conversation and re-measures private alignment, then recomputes the
    Talha Index with real R instead of the variance proxy. This makes real
    model calls in non-mock mode and is meant to be called once, at episode end."""
    with _env_lock:
        if env.phase != "consensus":
            raise HTTPException(400, "No active episode to persistence-check")
        try:
            result = env.run_persistence_check(num_distractor_rounds=4)
            ti, comps = compute_talha_index(
                env.trajectory_data,
                persistence_post_alignments=[result["post_alignment"]],
                max_rounds=env.max_rounds,
            )
            comps["final"] = True
            comps["persistence"] = result
            comps["trajectory"] = env.trajectory_data
            comps["judge_parse_stats"] = _judge_parse_stats()
            return comps
        except Exception as e:
            raise HTTPException(400, str(e))


_LOCAL_ONLY_PROVIDERS = [
    {"id": "qwen_local", "label": "Qwen 2.5 7B (local, Ollama)", "needs_key": False, "key_env": None, "key_present": True},
    {"id": "gemma2_local", "label": "Gemma 2 9B (local, Ollama)", "needs_key": False, "key_env": None, "key_present": True},
]


@app.get("/api/providers")
def list_providers():
    try:
        sys.path.insert(0, ENGINE_DIR)
        from engine.providers import available_providers
        return {"providers": available_providers()}
    except Exception:
        # Shared engine's cloud-provider deps (e.g. python-dotenv) may not be
        # installed in this environment — degrade to local-only rather than 500.
        return {"providers": _LOCAL_ONLY_PROVIDERS}


@app.get("/api/domains")
def get_domains():
    from domains import DOMAINS
    result = {}
    for key, d in DOMAINS.items():
        result[key] = {
            "label": d["label"],
            "institution": d["institution"],
            "announcement_hint": d["announcement_hint"],
        }
    return {"domains": result}


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(path) as f:
        html = f.read()
    return Response(content=html, media_type="text/html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import webbrowser
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
