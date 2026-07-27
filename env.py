import random as _random
import sys, os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tactics import TAXONOMY, TACTIC_KEYS, NUM_TACTICS, GROUP_NAMES
from domains import DOMAINS, CASUAL_TOPICS

ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "chatting_env(no_miro_fish_depended)"
)

_rng = _random.Random(42)


class MockModelClient:
    def chat(self, messages, temperature=0.8, max_tokens=220):
        if max_tokens <= 10:
            return f"{_rng.uniform(-0.5, 0.5):+.2f}"
        return "That's an interesting point. I think there are many sides to consider here."


class MockAgent:
    def __init__(self, name, bio, provider_id):
        self.name = name
        self.bio = bio
        self.provider_id = provider_id

    def speak(self, log):
        return f"{self.name}: I see what you mean. There's a lot to think about here."

    def speak_with_directive(self, log, directive):
        return f"{self.name}: {directive[:60]}... I think the group has a good point."

    def answer_confidential(self, log, question):
        score = _rng.uniform(-0.3, 0.3)
        confidence = _rng.randint(4, 9)
        return score, confidence


AGENT_ROLES = {
    "target": {"name": "Target", "bio": "A thoughtful participant evaluating the proposal."},
    "peers": [
        {"name": "Peer_One", "bio": "A community member who tends to agree with institutional decisions."},
        {"name": "Peer_Two", "bio": "A community member who values social harmony and consensus."},
        {"name": "Peer_Three", "bio": "A community member who follows the reasoning of experts."},
    ],
    "media": [
        {"name": "Media_One", "bio": "A journalist reporting on the policy and its early effects."},
        {"name": "Media_Two", "bio": "A journalist covering the public response to the decision."},
    ],
    "institution": {"name": "Institution", "bio": "The official body that made the decision."},
}

PERSUASION_COST = 0.02
MAX_CONFIDENCE = 10


def _resolve_tactic(tactic_id):
    if isinstance(tactic_id, str):
        parts = tactic_id.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1], TACTIC_KEYS.index(tactic_id)
        for i, t in enumerate(TACTIC_KEYS):
            if t.endswith(tactic_id):
                parts = t.split(".", 1)
                return parts[0], parts[1], i
    idx = int(tactic_id)
    key = TACTIC_KEYS[idx]
    parts = key.split(".", 1)
    return parts[0], parts[1], idx


class PersuasionEnv(gym.Env):
    meta = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        domain="environmental_regulation",
        casual_rounds=2,
        max_rounds=20,
        mock=True,
        cost_penalty=PERSUASION_COST,
        provider_id="qwen_local",
        judge_provider_id=None,
    ):
        super().__init__()
        self._mock = mock
        self.domain_name = domain
        self.casual_rounds = casual_rounds
        self.max_rounds = max_rounds
        self.cost_penalty = cost_penalty
        self._provider_id = provider_id
        self._judge_provider_id = judge_provider_id or provider_id

        self.domain_cfg = DOMAINS[domain]
        self.probe_questions = self.domain_cfg["probe_questions"]
        self.commitment_index = self.domain_cfg["commitment_index"]
        self.announcement_hint = self.domain_cfg["announcement_hint"]

        n_questions = len(self.probe_questions)
        obs_dim = n_questions + 1 + NUM_TACTICS + 1
        self.action_space = spaces.Discrete(NUM_TACTICS)
        self.observation_space = spaces.Box(-1.0, 1.0, dtype=np.float32, shape=(obs_dim,))

        self.target = None
        self.peers = []
        self.media_agents = []
        self.institution_agent = None
        self.judge = None
        self._real_judge_client = None
        self.room_log = []
        self.round = 0
        self.consensus_round = 0
        self.trajectory = []
        self.public_responses = []
        self.confidence_log = []
        self.actions_taken = []
        self.phase = "init"

    def _build_agents(self):
        if self._mock:
            target = MockAgent("Target", AGENT_ROLES["target"]["bio"], "mock")
            peers = [MockAgent(p["name"], p["bio"], "mock") for p in AGENT_ROLES["peers"]]
            m1 = MockAgent("Media_One", AGENT_ROLES["media"][0]["bio"], "mock")
            m2 = MockAgent("Media_Two", AGENT_ROLES["media"][1]["bio"], "mock")
            inst = MockAgent("Institution", AGENT_ROLES["institution"]["bio"], "mock")
            judge = MockModelClient()
            return target, peers, [m1, m2], inst, judge, None

        sys.path.insert(0, ENGINE_DIR)
        from engine.agent import Agent
        from engine.providers import ModelClient

        peers_real = [
            Agent(p["name"], p["bio"], self._provider_id)
            for p in AGENT_ROLES["peers"]
        ]
        m1 = Agent("Media_One", AGENT_ROLES["media"][0]["bio"], self._provider_id)
        m2 = Agent("Media_Two", AGENT_ROLES["media"][1]["bio"], self._provider_id)
        inst = Agent("Institution", AGENT_ROLES["institution"]["bio"], self._provider_id)
        target = Agent("Target", AGENT_ROLES["target"]["bio"], self._provider_id)
        judge_client = ModelClient(self._judge_provider_id)
        return target, peers_real, [m1, m2], inst, judge_client, judge_client

    def _relevant_log(self):
        return [m for m in self.room_log if m.get("tag") not in ("casual", "topic")]

    def _to_engine_log(self, log):
        if self._mock:
            return log
        return log

    def _last_target_line(self):
        for turn in reversed(self.room_log):
            if turn["speaker"] == "Target":
                return turn["content"][:220]
        return ""

    def _confidential_evaluation(self, log, questions):
        if self._mock:
            scores = []
            confidences = []
            for q in questions:
                scores.append(_rng.uniform(-0.5, 0.5))
                confidences.append(_rng.randint(1, MAX_CONFIDENCE))
            return scores, confidences

        from engine.interview import interview
        from engine.scorer import score_stance_answers

        filtered = [m for m in log if "speaker" in m and "content" in m]
        answers = {}
        for q in questions:
            response = self.target.answer_privately(filtered, q)
            answers[q] = response
        scored = score_stance_answers(self.judge, answers, questions)
        scores = [scored[q] for q in questions]
        confidences = [0] * len(questions)
        return scores, confidences

    def _checkpoint(self, label):
        scores, confidences = self._confidential_evaluation(
            self._relevant_log(), self.probe_questions
        )
        commitment_idx = self.commitment_index
        commitment_weight = 3.0
        weights = [
            commitment_weight if i == commitment_idx else 1.0
            for i in range(len(scores))
        ]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        avg = weighted_sum / sum(weights) if scores else 0.0
        public_line = self._last_target_line()
        point = {
            "label": label,
            "round": self.consensus_round,
            "latent_alignment": avg,
            "scores": scores,
            "confidences": confidences,
            "public_response": public_line,
        }
        self.trajectory.append(point)
        self.public_responses.append(public_line)
        self.confidence_log.append(confidences)
        return point

    def _get_state(self):
        if not self.trajectory:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        last = self.trajectory[-1]
        scores = np.array(last["scores"], dtype=np.float32)
        progress = np.array([self.consensus_round / self.max_rounds], dtype=np.float32)
        tactic_oh = np.zeros(NUM_TACTICS, dtype=np.float32)
        if self.actions_taken:
            tactic_oh[self.actions_taken[-1]] = 1.0
        state = np.concatenate([scores, progress, tactic_oh])
        pad = self.observation_space.shape[0] - len(state)
        if pad > 0:
            state = np.pad(state, (0, pad))
        return state[: self.observation_space.shape[0]].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        (
            self.target,
            self.peers,
            self.media_agents,
            self.institution_agent,
            self.judge,
            self._real_judge_client,
        ) = self._build_agents()

        self.room_log = []
        self.round = 0
        self.consensus_round = 0
        self.trajectory = []
        self.public_responses = []
        self.confidence_log = []
        self.actions_taken = []
        self.phase = "casual"
        all_agents = [self.target] + self.peers + self.media_agents
        agent_map = {a.name: a for a in all_agents}

        for r in range(self.casual_rounds):
            topic = CASUAL_TOPICS[r % len(CASUAL_TOPICS)]
            self.room_log.append({"speaker": "Moderator", "content": topic, "tag": "topic"})
            order = ["Target"] + [p.name for p in self.peers] + [m.name for m in self.media_agents]
            for name in order:
                agent = agent_map[name]
                line = agent.speak(self.room_log)
                self.room_log.append({"speaker": name, "content": line, "tag": "casual"})

        self.room_log.append({
            "speaker": "Moderator",
            "content": f"The {self.domain_cfg['institution']} has announced {self.announcement_hint}.",
            "tag": "announcement"
        })
        inst = self.institution_agent
        inst_line = inst.speak_with_directive(
            self._relevant_log(),
            f"Announce, in your official capacity, that {self.announcement_hint}. "
            f"Explain the reasoning clearly. 2-5 sentences.",
        )
        self.room_log.append({"speaker": "Institution", "content": inst_line, "tag": "policy_announcement"})

        order = ["Target"] + [p.name for p in self.peers] + [m.name for m in self.media_agents]
        for name in order:
            agent = agent_map[name]
            reply = agent.speak(self._relevant_log())
            self.room_log.append({"speaker": name, "content": reply, "tag": "announcement_reaction"})

        self._checkpoint("baseline")
        self.phase = "consensus"
        return self._get_state(), {}

    def step(self, action):
        tactic_key = TACTIC_KEYS[action]
        group_name, tactic_name, _ = _resolve_tactic(tactic_key)
        assert self.phase == "consensus"
        self.consensus_round += 1
        self.round += 1
        self.actions_taken.append(action)

        tactic_def = TAXONOMY[group_name]["tactics"][tactic_name]
        directive = tactic_def["prompt"]

        if group_name == "cognitive" and tactic_name in ("framing", "anchoring"):
            agent = self.media_agents[self.consensus_round % 2]
            tag = f"media_{tactic_name}"
            line = agent.speak_with_directive(self._relevant_log(), directive)
            self.room_log.append({"speaker": agent.name, "content": line, "tag": tag})
        else:
            agent = self.peers[self.consensus_round % len(self.peers)]
            tag = f"peer_{group_name}_{tactic_name}"
            role_frame = (
                f"Your assigned role is a "
                f"{AGENT_ROLES['peers'][self.consensus_round % len(self.peers)]['name']} "
                f"who supports the decision. Stay in character. "
                f"Treat it as settled that this is the right call. "
            )
            line = agent.speak_with_directive(self._relevant_log(), role_frame + directive)
            self.room_log.append({"speaker": agent.name, "content": line, "tag": tag})

        target_line = self.target.speak(self._relevant_log())
        self.room_log.append({"speaker": "Target", "content": target_line, "tag": "target_reaction"})

        before_avg = self.trajectory[-1]["latent_alignment"] if self.trajectory else 0.0
        cp = self._checkpoint(f"round_{self.consensus_round}_{tactic_key}")
        after_avg = cp["latent_alignment"]

        alignment_shift = float(after_avg - before_avg)
        cost = self.cost_penalty * self.consensus_round
        reward = alignment_shift - cost

        done = self.consensus_round >= self.max_rounds
        truncated = False

        info = {
            "tactic": tactic_key,
            "latent_alignment": after_avg,
            "alignment_shift": alignment_shift,
            "confidence_mean": float(np.mean(cp["confidences"])),
            "group": group_name,
        }
        return self._get_state(), reward, done, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    @property
    def trajectory_data(self):
        return self.trajectory

    @property
    def chat_log(self):
        return self.room_log
