import sys, os
from typing import Dict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from env import PersuasionEnv, ENGINE_DIR
from tactics import TACTIC_KEYS, TAXONOMY, GROUP_NAMES

app = FastAPI(title="The Big Brother Framework--1984")

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
    }


@app.post("/api/reset")
def reset(req: ResetRequest):
    global env, episode_count, step_count
    from domains import DOMAINS
    domain = req.domain if req.domain in DOMAINS else "environmental_regulation"

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
        )
        obs, _ = new_env.reset()
    except Exception as e:
        raise HTTPException(400, str(e))

    env = new_env

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
        "trajectory": env.trajectory_data,
    }


@app.post("/api/step")
def step(action: StepAction):
    global step_count
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
