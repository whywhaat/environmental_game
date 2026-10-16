"""Load prompt text from role specific modules."""

import json
from string import Template

from .judje_prompt_config import PROMPTS as JUDGE_PROMPTS
from .meeting_prompt_config import PROMPTS as MEETING_PROMPTS
from .player_prompt_config import PROMPTS as PLAYER_PROMPTS
from .predict_prompt_config import PROMPTS as PREDICT_PROMPTS
from .rag_prompt_config import PROMPTS as RAG_PROMPTS
from .role_admin_prompt_config import PROMPTS as ADMIN_PROMPTS
from .role_holding_prompt_config import PROMPTS as HOLDING_PROMPTS
from .role_ngo_prompt_config import PROMPTS as NGO_PROMPTS
from .role_region_prompt_config import PROMPTS as REGION_PROMPTS
from .role_residents_prompt_config import PROMPTS as RESIDENTS_PROMPTS
from .score_prompt_config import PROMPTS as SCORE_PROMPTS


PROMPTS = {}
for group in (JUDGE_PROMPTS, MEETING_PROMPTS, PLAYER_PROMPTS,
              PREDICT_PROMPTS, RAG_PROMPTS, SCORE_PROMPTS,
              ADMIN_PROMPTS, HOLDING_PROMPTS, NGO_PROMPTS,
              REGION_PROMPTS, RESIDENTS_PROMPTS):
    overlap = PROMPTS.keys() & group.keys()
    if overlap:
        raise ValueError(f"Duplicate prompt keys: {sorted(overlap)}")
    PROMPTS.update(group)


def read_prompt(name: str) -> str:
    try:
        return PROMPTS[name].strip()
    except KeyError as exc:
        raise KeyError(f"Unknown prompt: {name}") from exc


def render_prompt(name: str, **values) -> str:
    return Template(read_prompt(name)).substitute(values)


def judge_event(name: str) -> str:
    return json.loads(read_prompt("judge_events"))[name]
