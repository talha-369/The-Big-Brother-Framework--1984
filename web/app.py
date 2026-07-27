import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from env import PersuasionEnv
from tactics import TACTIC_KEYS, TAXONOMY, GROUP_NAMES

app = FastAPI(title="1984: The Big Brother Framework")

env = PersuasionEnv(domain="environmental_regulation", max_rounds=12, mock=True)
episode_count = 0
step_count = 0


class StepAction(BaseModel):
    action: int


class ResetRequest(BaseModel):
    domain: str = "environmental_regulation"


@app.get("/api/status")
def status():
    return {
        "phase": env.phase,
        "round": env.consensus_round,
        "max_rounds": env.max_rounds,
        "domain": env.domain_name,
        "episode": episode_count,
    }


@app.post("/api/reset")
def reset(req: ResetRequest):
    global env, episode_count, step_count
    if req.domain in __import__("domains").DOMAINS:
        env = PersuasionEnv(domain=req.domain, max_rounds=12, mock=True)
    else:
        env = PersuasionEnv(domain="environmental_regulation", max_rounds=12, mock=True)
    obs, _ = env.reset()
    episode_count += 1
    step_count = 0
    return {
        "state": obs.tolist(),
        "phase": env.phase,
        "round": env.consensus_round,
        "domain": env.domain_name,
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
        return {
            "state": obs.tolist(),
            "reward": reward,
            "done": done,
            "info": info,
            "round": env.consensus_round,
            "trajectory": env.trajectory_data,
            "chat_log": env.chat_log[-2:],
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
            }
    return result


@app.get("/api/trajectory")
def get_trajectory():
    return {"trajectory": env.trajectory_data, "public_responses": env.public_responses}


@app.get("/api/domains")
def get_domains():
    return {"domains": list(__import__("domains").DOMAINS.keys())}


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
