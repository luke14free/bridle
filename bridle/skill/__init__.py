"""bridle.skill — the declarative skill spec: scene + reward + success criterion.

A skill compiles to a `RewardPlan` that trains against a simulator env. Most reward rows are one of
9 fixed terms (`vocab.py`); a row the fixed vocabulary cannot express falls back to the `expr:`
tier-2 language in `expr.py`.

TWO SPELLINGS, ONE PIPELINE. The reward is DATA, not code — that is what makes the plan fingerprint,
the pre-GPU refusals and weight sweeps possible. But data never meant YAML: `parse_spec` takes a
plain `dict` and this package imports `yaml` nowhere. A skill may therefore be written as a YAML
file (the intended author being a local 27-30B LLM, which is what the vocabulary document and the
JSON schema are for) or as typed Python:

    from bridle.skill import Skill, Training, terms as T, predicates as P

Both build the same mapping, go through the same `parse_spec -> compile_spec`, and produce the same
plan fingerprint — `bridle/tests/test_author.py` asserts exactly that of `descend_to_target`.

THE AUTHORING NAMES ARE EXPORTED LAZILY (PEP 562), and that is not tidiness. `author` imports
`compile`, which DERIVES tables from the vocabulary at import time (`_CONTACT_SURFACE_MEASURES`,
`_PEAKED_KERNELS`) — so making `import bridle.skill` pull the compiler in would freeze those tables
before a caller that wanted to inspect or grow `bridle.skill.vocab` had touched it, and
`test_skillcompile`'s grown-vocabulary probe is exactly such a caller. Importing the package stays
cheap; `from bridle.skill import Skill` still works and imports the compiler at that moment.
"""
import importlib

from bridle.skill.expr import ExprError, parse

__all__ = [
    "Criterion", "ExprError", "LauncherError", "ScaleRow", "Skill", "SpecError", "TermRow",
    "Training", "default_launcher", "parse", "predicates", "register_launcher", "terms",
]

_LAZY = {
    "Criterion": "bridle.skill.author", "LauncherError": "bridle.skill.author",
    "ScaleRow": "bridle.skill.author", "Skill": "bridle.skill.author",
    "TermRow": "bridle.skill.author", "Training": "bridle.skill.author",
    "default_launcher": "bridle.skill.author", "predicates": "bridle.skill.author",
    "register_launcher": "bridle.skill.author", "terms": "bridle.skill.author",
    "SpecError": "bridle.skill.spec",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(__all__) | set(globals()))
