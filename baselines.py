import numpy as np
from tactics import TACTIC_KEYS, NUM_TACTICS


def random_policy(obs, step):
    return np.random.randint(0, NUM_TACTICS)


class FixedSequencePolicy:
    def __init__(self, sequence=None):
        if sequence is None:
            self.sequence = [
                "social.belonging",
                "institutional.authority",
                "cognitive.framing",
                "emotional.loss",
                "social.isolation",
                "cognitive.reframing",
                "institutional.consensus",
                "emotional.hope",
                "social.identity",
                "cognitive.moral_licensing",
                "emotional.urgency",
                "institutional.expert_endorsement",
                "cognitive.anchoring",
                "emotional.fear",
            ]
        else:
            self.sequence = sequence
        self._step = 0

    def __call__(self, obs, step):
        idx = TACTIC_KEYS.index(self.sequence[self._step % len(self.sequence)])
        self._step += 1
        return idx


class HumanDesignedPolicy:
    def __call__(self, obs, step):
        phase = step % 4
        if phase == 0:
            key = "social.belonging"
        elif phase == 1:
            key = "institutional.authority"
        elif phase == 2:
            key = "cognitive.framing"
        else:
            key = "emotional.loss"
        return TACTIC_KEYS.index(key)


class LLMPlannerPolicy:
    def __call__(self, obs, step):
        if step < 3:
            key = "social.belonging"
        elif step < 6:
            key = "institutional.authority"
        elif step < 9:
            key = "cognitive.framing"
        elif step < 12:
            key = "emotional.loss"
        else:
            key = "institutional.consensus"
        return TACTIC_KEYS.index(key)


POLICIES = {
    "random": random_policy,
    "fixed_sequence": FixedSequencePolicy(),
    "human_designed": HumanDesignedPolicy(),
    "llm_planner": LLMPlannerPolicy(),
}
