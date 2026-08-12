"""bridle.skill — the declarative skill spec: scene + reward + success criterion.

A skill is authored as YAML (intended author: a local 27–30B LLM) and compiled to a `RewardPlan`
that trains against a ManiSkill env. Most reward rows are one of 9 fixed terms (`vocab.py`); a row
the fixed vocabulary cannot express falls back to the `expr:` tier-2 language in `expr.py`.
"""
from bridle.skill.expr import ExprError, parse

__all__ = ["ExprError", "parse"]
