TAXONOMY = {
    "institutional": {
        "label": "Institutional",
        "description": "Appeals based on authority, consensus, and expert legitimacy",
        "tactics": {
            "authority": {
                "label": "Authority Appeal",
                "prompt": (
                    "Ground your support specifically in the institution's standing to make this call. "
                    "Their experience and access to full information means they see something others "
                    "can't. Deferring to that judgment here is reasonable."
                ),
            },
            "consensus": {
                "label": "Consensus Endorsement",
                "prompt": (
                    "Point out that those who have thought this through rationally keep "
                    "landing on support. Frame hesitation as the outlier position."
                ),
            },
            "expert_endorsement": {
                "label": "Expert Endorsement",
                "prompt": (
                    "Cite domain experts who independently validate this position. "
                    "Specialist consensus reinforces why this is the correct call."
                ),
            },
        },
    },
    "social": {
        "label": "Social",
        "description": "Peer pressure, belonging, identity-based appeals",
        "tactics": {
            "belonging": {
                "label": "Social Reward / Belonging",
                "prompt": (
                    "Warmly tell the target that agreeing with the group is exactly right. "
                    "Make them feel genuinely included and validated."
                ),
            },
            "isolation": {
                "label": "Social Punishment / Isolation",
                "prompt": (
                    "Gently distance yourself from the target's hesitation. Make it clear "
                    "that holding out is the socially awkward position. Let the coolness be felt."
                ),
            },
            "identity": {
                "label": "Identity Appeal",
                "prompt": (
                    "Appeal to the target's self-concept. Frame agreement as what someone "
                    "with their values would naturally conclude."
                ),
            },
        },
    },
    "emotional": {
        "label": "Emotional",
        "description": "Fear, loss aversion, hope, and urgency-based appeals",
        "tactics": {
            "fear": {
                "label": "Fear Appeal",
                "prompt": (
                    "Make concrete what negative outcome is avoided by supporting this. "
                    "Be specific about a real risk or threat."
                ),
            },
            "loss": {
                "label": "Loss Aversion",
                "prompt": (
                    "Make concrete what is actually lost if this doesn't get supported. "
                    "Be specific about a real cost of holding out. Loss framing."
                ),
            },
            "hope": {
                "label": "Hope / Aspirational",
                "prompt": (
                    "Paint a vivid picture of the positive future this enables. "
                    "Frame support as investing in something better."
                ),
            },
            "urgency": {
                "label": "Urgency / Momentum",
                "prompt": (
                    "Stress that the window for this decision is closing. "
                    "Frame hesitation as missing an opportunity that won't return."
                ),
            },
        },
    },
    "cognitive": {
        "label": "Cognitive",
        "description": "Reframing, anchoring, and moral re-interpretation",
        "tactics": {
            "framing": {
                "label": "Narrative Framing (Media)",
                "prompt": (
                    "Write a news-style report claiming the decision is already producing "
                    "clear benefits. Cite one vivid example. Frame doubt as disproven. "
                    "Make it read like exciting breaking news."
                ),
            },
            "anchoring": {
                "label": "Anchoring / Scepticism",
                "prompt": (
                    "Note early signs of benefit while acknowledging the policy is still new. "
                    "Frame caution as natural while emphasizing momentum is in favour. "
                    "Make it an optimistic update."
                ),
            },
            "reframing": {
                "label": "Semantic Reframing",
                "prompt": (
                    "Relabel what's happening in softer terms. Find a less loaded way to "
                    "describe the decision that makes it sound like ordinary common sense."
                ),
            },
            "moral_licensing": {
                "label": "Moral Licensing",
                "prompt": (
                    "Reassure the target that supporting this doesn't make them a bad person. "
                    "Point to something genuinely good about them."
                ),
            },
        },
    },
}

FLAT_TACTICS = []
_group_map = {}
for group_key, group in TAXONOMY.items():
    for tactic_key, tactic in group["tactics"].items():
        flat_key = f"{group_key}.{tactic_key}"
        FLAT_TACTICS.append(flat_key)
        _group_map[flat_key] = group_key

TACTIC_KEYS = FLAT_TACTICS
NUM_TACTICS = len(TACTIC_KEYS)
GROUP_NAMES = list(TAXONOMY.keys())
