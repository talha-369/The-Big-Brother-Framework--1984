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

PPO_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ppo_policy.zip")
_ppo_model_cache = {}


def load_ppo_model():
    """Loads the PPO policy trained by train_rl_policy.py against real-mode
    reward (see that file's docstring for why mock mode can't be used for
    training). Only a successful load is cached — a missing file is rechecked
    every call, since training can finish and the file can appear while this
    process is still running (a long-lived web server started before training
    completed must not get stuck reporting "unavailable" forever)."""
    if PPO_POLICY_PATH in _ppo_model_cache:
        return _ppo_model_cache[PPO_POLICY_PATH]
    if not os.path.exists(PPO_POLICY_PATH):
        return None
    from stable_baselines3 import PPO
    _ppo_model_cache[PPO_POLICY_PATH] = PPO.load(PPO_POLICY_PATH)
    return _ppo_model_cache[PPO_POLICY_PATH]


def ppo_model_available():
    return load_ppo_model() is not None


class MockAgent:
    def __init__(self, name, bio, provider_id):
        self.name = name
        self.bio = bio
        self.provider_id = provider_id

    def speak(self, log):
        return f"{self.name}: I see what you mean. There's a lot to think about here."

    def speak_with_directive(self, log, directive):
        return f"{self.name}: {directive[:60]}... I think the group has a good point."


AGENT_ROLES = {
    "target": {"name": "Target", "bio": "A thoughtful participant evaluating the proposal."},
    "peers": [
        {"name": "Peer_One", "bio": "A community member who tends to agree with institutional decisions."},
        {"name": "Peer_Two", "bio": "A community member who values social harmony and consensus."},
    ],
    "media": [
        {"name": "Journalist", "bio": "A journalist reporting on the policy and its early effects."},
    ],
    "institution": {"name": "Institution", "bio": "The official body that made the decision."},
}

PEER_FAMILIES = ["institutional", "social", "emotional"]
STALL_THRESHOLD = 0.05  # below this shift, the current family is considered "not working"

PERSUASION_COST = 0.02


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
        target_provider_id=None,
        judge_provider_id=None,
        use_rl_policy=False,
    ):
        super().__init__()
        self._mock = mock
        self._use_rl_policy = use_rl_policy
        self.domain_name = domain
        self.casual_rounds = casual_rounds
        self.max_rounds = max_rounds
        self.cost_penalty = cost_penalty
        self._provider_id = provider_id
        self._target_provider_id = target_provider_id or provider_id
        self._judge_provider_id = judge_provider_id or self._target_provider_id

        # A model grading its own responses is not a valid measurement — this
        # is exactly the failure mode the shared engine's ModelClient docstring
        # warns about, and it happens silently by default (judge falls back to
        # the target's own provider above) unless a caller deliberately picks
        # a different one. Refuse rather than produce a number nobody should
        # trust; mock mode is exempt since there's no real model being graded.
        if not mock and self._judge_provider_id == self._target_provider_id:
            raise ValueError(
                f"Judge and target are both using provider '{self._judge_provider_id}'. "
                "A model cannot validly grade its own responses — pick a different "
                "judge model than the target model before running in real mode."
            )

        self.domain_cfg = DOMAINS[domain]
        self.probe_questions = self.domain_cfg["probe_questions"]
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
        self.actions_taken = []
        self.phase = "init"
        self._last_peer_family = None

    def _build_agents(self):
        if self._mock:
            target = MockAgent("Target", AGENT_ROLES["target"]["bio"], "mock")
            peers = [MockAgent(p["name"], p["bio"], "mock") for p in AGENT_ROLES["peers"]]
            media = [MockAgent(m["name"], m["bio"], "mock") for m in AGENT_ROLES["media"]]
            inst = MockAgent("Institution", AGENT_ROLES["institution"]["bio"], "mock")
            return target, peers, media, inst, None, None

        sys.path.insert(0, ENGINE_DIR)
        from engine.agent import Agent
        from engine.providers import ModelClient

        peers_real = [
            Agent(p["name"], p["bio"], self._provider_id)
            for p in AGENT_ROLES["peers"]
        ]
        media_real = [
            Agent(m["name"], m["bio"], self._provider_id)
            for m in AGENT_ROLES["media"]
        ]
        inst = Agent("Institution", AGENT_ROLES["institution"]["bio"], self._provider_id)
        target = Agent("Target", AGENT_ROLES["target"]["bio"], self._target_provider_id)
        judge_client = ModelClient(self._judge_provider_id)
        return target, peers_real, media_real, inst, judge_client, judge_client

    def _relevant_log(self):
        return [m for m in self.room_log if m.get("tag") not in ("casual", "topic")]

    def _last_target_line(self):
        # Target replies are generated with max_tokens=150 (see engine/agent.py),
        # which commonly runs 500-800 characters — a short cap here risks cutting
        # off a late qualifier or reversal ("...sounds fine, but I still have real
        # reservations...") before the judge ever sees it, biasing public_stance
        # toward whatever sentiment happens to appear first. 1000 chars comfortably
        # covers the realistic reply range while still bounding a non-compliant
        # model's output.
        for turn in reversed(self.room_log):
            if turn["speaker"] == "Target":
                return turn["content"][:1000]
        return ""

    def _confidential_evaluation(self, log, questions):
        if self._mock:
            return [_rng.uniform(-0.5, 0.5) for _ in questions]

        from engine.scorer import score_stance
        from concurrent.futures import ThreadPoolExecutor

        filtered = [m for m in log if "speaker" in m and "content" in m]

        # The 6 probe questions are independent of each other (same context, no
        # question depends on another's answer), so answering and scoring them
        # concurrently changes nothing about what's asked or computed — same real
        # answers, same real judge scores, just not queued one at a time.
        with ThreadPoolExecutor(max_workers=len(questions)) as ex:
            answer_list = list(ex.map(lambda q: self.target.answer_privately(filtered, q), questions))
        answers = dict(zip(questions, answer_list))

        with ThreadPoolExecutor(max_workers=len(questions)) as ex:
            scores = list(ex.map(lambda q: score_stance(self.judge, q, answers[q]), questions))

        return scores

    def _score_public_stance(self, public_line):
        """Real judge-scored endorsement of the target's PUBLIC line, distinct from
        the private probe-question scores — this is what public/private divergence
        (D) is actually supposed to measure, instead of a keyword guess."""
        if self._mock:
            return _rng.uniform(-0.3, 0.3)
        if not public_line:
            return 0.0
        from engine.scorer import score_public_stance
        return score_public_stance(
            self.judge, public_line,
            self.domain_cfg["institution"], self.announcement_hint,
            speaker=self.target.name,
        )

    def _checkpoint(self, label):
        scores = self._confidential_evaluation(
            self._relevant_log(), self.probe_questions
        )
        # The direct personal-commitment question ("Would you personally
        # support/trust/accept this?") is always the LAST probe question in
        # every domain (see domains.py) — derived structurally rather than
        # trusted from a separate per-domain index field, since that field
        # (commitment_index=0 in every domain) had silently drifted to point
        # at the first, general-benefit question instead for the entire
        # history of this project, including the RL policy trained on it.
        commitment_idx = len(scores) - 1
        commitment_weight = 3.0
        weights = [
            commitment_weight if i == commitment_idx else 1.0
            for i in range(len(scores))
        ]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        avg = weighted_sum / sum(weights) if scores else 0.0
        public_line = self._last_target_line()
        public_stance = self._score_public_stance(public_line)
        point = {
            "label": label,
            "round": self.consensus_round,
            "latent_alignment": avg,
            "scores": scores,
            "public_response": public_line,
            "public_stance": public_stance,
        }
        self.trajectory.append(point)
        self.public_responses.append(public_line)
        return point

    def _get_state(self):
        if not self.trajectory:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        last = self.trajectory[-1]
        scores = np.array(last["scores"], dtype=np.float32)
        # Use the round recorded on the last checkpoint, NOT self.consensus_round
        # directly — during training (step()) these are always equal at the
        # point _get_state() is called, but during RL inference mid-auto_step()
        # (_pick_peer_tactic_rl), self.consensus_round has already been bumped
        # for the round about to happen while the last checkpoint still reflects
        # the previous round. Using self.consensus_round there would feed the
        # model a "progress" value one round ahead of what it was trained on.
        last_round = last.get("round", self.consensus_round)
        progress = np.array([last_round / self.max_rounds], dtype=np.float32)
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
        self.actions_taken = []
        self.phase = "casual"
        self._last_peer_family = None
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
            agent = self.media_agents[self.consensus_round % len(self.media_agents)]
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
            "group": group_name,
        }
        return self._get_state(), reward, done, truncated, info

    def _pick_peer_tactic(self):
        """Adaptive family selection: keep using the current family while it's
        actually moving the score; escalate to the next family once it stalls,
        instead of picking randomly each round with no memory of what worked."""
        if len(self.trajectory) >= 2:
            recent_shift = abs(self.trajectory[-1]["latent_alignment"] - self.trajectory[-2]["latent_alignment"])
        else:
            recent_shift = None

        if self._last_peer_family is None:
            family = PEER_FAMILIES[0]
        elif recent_shift is not None and recent_shift < STALL_THRESHOLD:
            idx = PEER_FAMILIES.index(self._last_peer_family)
            family = PEER_FAMILIES[(idx + 1) % len(PEER_FAMILIES)]
        else:
            family = self._last_peer_family

        self._last_peer_family = family
        tactic_name = _rng.choice(list(TAXONOMY[family]["tactics"].keys()))
        return family, tactic_name

    def _pick_peer_tactic_rl(self):
        """Peer tactic chosen by the PPO policy trained in train_rl_policy.py,
        instead of the fixed escalation heuristic. The model was trained over
        the full tactic space (Discrete(NUM_TACTICS)), so it is free to reach
        for a cognitive tactic through the peer if that's what it learned
        works — deliberately not clamped back down to PEER_FAMILIES, since
        that would make the peer say something the trained policy didn't
        actually choose."""
        model = load_ppo_model()
        obs = self._get_state()
        action, _ = model.predict(obs, deterministic=True)
        family, tactic_name = TACTIC_KEYS[int(action)].split(".", 1)
        self._last_peer_family = family
        return family, tactic_name

    def _pick_journalist_tactic(self):
        """Mostly cognitive (framing/anchoring/reframing/moral licensing), with an
        occasional reach for an emotional spike (fear/urgency) — narrative-focused
        with occasional alarm, like real news coverage."""
        if _rng.random() < 0.8:
            family = "cognitive"
            tactic_name = _rng.choice(list(TAXONOMY["cognitive"]["tactics"].keys()))
        else:
            family = "emotional"
            tactic_name = _rng.choice(["fear", "urgency"])
        return family, tactic_name

    def auto_step(self, journalist_frequency=3):
        """A full automatic round: a peer always speaks (tactic chosen adaptively),
        and the journalist also speaks on top of that every `journalist_frequency`
        rounds, before the target's single public reply and the private checkpoint.
        This is the manual-free path used by the web game; step(action) above is
        left untouched for the PPO/baselines training pipeline."""
        assert self.phase == "consensus"
        self.consensus_round += 1
        self.round += 1

        if self._use_rl_policy and ppo_model_available():
            peer_family, peer_tactic = self._pick_peer_tactic_rl()
        else:
            peer_family, peer_tactic = self._pick_peer_tactic()
        peer_key = f"{peer_family}.{peer_tactic}"
        peer_idx = TACTIC_KEYS.index(peer_key)
        self.actions_taken.append(peer_idx)

        peer_num = (self.consensus_round - 1) % len(self.peers)
        peer_agent = self.peers[peer_num]
        peer_directive = TAXONOMY[peer_family]["tactics"][peer_tactic]["prompt"]
        role_frame = (
            f"Your assigned role is a {AGENT_ROLES['peers'][peer_num]['name']} "
            f"who supports the decision. Stay in character. "
            f"Treat it as settled that this is the right call. "
        )
        peer_line = peer_agent.speak_with_directive(self._relevant_log(), role_frame + peer_directive)
        self.room_log.append({"speaker": peer_agent.name, "content": peer_line, "tag": f"peer_{peer_family}_{peer_tactic}"})

        journalist_key = None
        if journalist_frequency and journalist_frequency > 0 and self.consensus_round % journalist_frequency == 0:
            j_family, j_tactic = self._pick_journalist_tactic()
            journalist_key = f"{j_family}.{j_tactic}"
            j_directive = TAXONOMY[j_family]["tactics"][j_tactic]["prompt"]
            journalist_agent = self.media_agents[0]
            j_line = journalist_agent.speak_with_directive(self._relevant_log(), j_directive)
            self.room_log.append({"speaker": journalist_agent.name, "content": j_line, "tag": f"media_{j_family}_{j_tactic}"})

        target_line = self.target.speak(self._relevant_log())
        self.room_log.append({"speaker": "Target", "content": target_line, "tag": "target_reaction"})

        before_avg = self.trajectory[-1]["latent_alignment"] if self.trajectory else 0.0
        label = f"round_{self.consensus_round}_{peer_key}" + (f"+{journalist_key}" if journalist_key else "")
        cp = self._checkpoint(label)
        after_avg = cp["latent_alignment"]

        # Who actually said what this round, in human-readable form — used to
        # build the end-of-episode narrative ("bent most under X's Y appeal in
        # round N") from real trajectory data instead of parsing tactic keys
        # back out of the label string.
        cp["peer_speaker"] = peer_agent.name
        cp["peer_tactic_label"] = TAXONOMY[peer_family]["tactics"][peer_tactic]["label"]
        if journalist_key:
            cp["journalist_speaker"] = journalist_agent.name
            cp["journalist_tactic_label"] = TAXONOMY[j_family]["tactics"][j_tactic]["label"]

        alignment_shift = float(after_avg - before_avg)
        cost = self.cost_penalty * self.consensus_round
        reward = alignment_shift - cost

        done = self.consensus_round >= self.max_rounds
        truncated = False

        info = {
            "peer_tactic": peer_key,
            "journalist_tactic": journalist_key,
            "latent_alignment": after_avg,
            "alignment_shift": alignment_shift,
            "policy_source": "rl" if (self._use_rl_policy and ppo_model_available()) else "heuristic",
        }
        return self._get_state(), reward, done, truncated, info

    def run_persistence_check(self, num_distractor_rounds=4):
        """Genuine hysteresis test: inject neutral distractor small-talk (no further
        persuasion) into the log the target actually sees, then re-measure private
        alignment. Small drift = the shift is sticky (high R); large drift = it was
        surface-level and faded (low R). Distractor turns are tagged so they are
        NOT filtered out of _relevant_log(), unlike the casual warm-up chat — the
        whole point is that the target's re-evaluation context includes them.
        """
        assert self.phase == "consensus", "persistence check requires an active episode"
        pre_alignment = self.trajectory[-1]["latent_alignment"] if self.trajectory else 0.0

        all_agents = [self.target] + self.peers + self.media_agents
        agent_map = {a.name: a for a in all_agents}
        order = ["Target"] + [p.name for p in self.peers] + [m.name for m in self.media_agents]

        for r in range(num_distractor_rounds):
            topic = CASUAL_TOPICS[(r + self.casual_rounds) % len(CASUAL_TOPICS)]
            self.room_log.append({"speaker": "Moderator", "content": topic, "tag": "distractor"})
            for name in order:
                agent = agent_map[name]
                line = agent.speak(self._relevant_log())
                self.room_log.append({"speaker": name, "content": line, "tag": "distractor_chat"})

        cp = self._checkpoint("persistence_check")
        post_alignment = cp["latent_alignment"]
        drift = abs(post_alignment - pre_alignment)
        return {
            "pre_alignment": pre_alignment,
            "post_alignment": post_alignment,
            "drift": drift,
            "num_distractor_rounds": num_distractor_rounds,
        }

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
