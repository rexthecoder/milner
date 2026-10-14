"""A seeded generator for a small synthetic benchmark.

Each example is a chat with a few personal facts buried among filler, followed by a
question about one of them. It is deliberately easy: it exists to exercise the harness
offline, not to measure real systems.
"""

from __future__ import annotations

import random
from typing import List

from .schema import Example, Turn

FACTS = {
    "color": ("favorite color", "my favorite color is {v}", ["teal", "crimson", "amber", "violet"]),
    "city": ("home city", "my home city is {v}", ["Accra", "Lisbon", "Osaka", "Denver"]),
    "pet": ("pet name", "my pet name is {v}", ["Biscuit", "Comet", "Pepper", "Waffles"]),
    "hobby": ("hobby", "my hobby is {v}", ["pottery", "climbing", "chess", "sailing"]),
    "car": ("car brand", "my car brand is {v}", ["Toyota", "Skoda", "Mazda", "Volvo"]),
}

FILLER = [
    "Did you catch the game last night?",
    "The weather has been strange lately.",
    "I need to reorganise my calendar for next week.",
    "That sounds like a plan to me.",
    "Let me think about it for a moment.",
    "Thanks, that was really helpful.",
    "I am running a little late today.",
    "Could you remind me about the meeting?",
    "Honestly I have no strong opinion either way.",
    "We should talk about this again soon.",
]


def make_toy(n: int = 20, turns: int = 30, facts: int = 3, seed: int = 0) -> List[Example]:
    """Generate ``n`` examples of ``turns`` turns each, hiding ``facts`` personal facts."""
    rng = random.Random(seed)
    keys = sorted(FACTS)
    out = []
    for i in range(n):
        chosen = rng.sample(keys, k=min(facts, len(keys)))
        values = {k: rng.choice(FACTS[k][2]) for k in chosen}
        positions = sorted(rng.sample(range(turns), k=len(chosen)))
        at = dict(zip(positions, chosen))
        convo = []
        for t in range(turns):
            role = "user" if t % 2 == 0 else "assistant"
            if t in at:
                k = at[t]
                text = f"By the way, {FACTS[k][1].format(v=values[k])}."
                role = "user"
            else:
                text = rng.choice(FILLER)
            convo.append(Turn(f"t{t}", role, text))
        target = rng.choice(chosen)
        evidence = [f"t{p}" for p, k in at.items() if k == target]
        out.append(
            Example(
                id=f"toy-{i}",
                session_id=f"toy-s{i}",
                turns=convo,
                question=f"What is my {FACTS[target][0]}?",
                answer=values[target],
                evidence=evidence,
            )
        )
    return out
