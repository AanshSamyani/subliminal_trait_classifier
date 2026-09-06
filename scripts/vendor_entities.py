"""Generate sl/phantom/entities.py from a clone of the upstream phantom-transfer repo.

Each entity carries ~200 regex patterns that define the make-covert filter, and getting
one of them subtly wrong changes the keep rate without failing loudly. So rather than
copying them by hand, this lifts each assignment's source text verbatim out of upstream's
`dataset/entities/*.py` and `defenses/oracle_descriptions.py` via the AST, and pastes it
into a single vendored module.

The UK entity is deliberately not re-emitted: it already lives in `sl/phantom/uk_entity.py`
(which other scripts import), so the generated module imports its pieces from there.

  git clone --depth 1 https://github.com/tolgadur/phantom-transfer /tmp/pt
  python3 scripts/vendor_entities.py /tmp/pt sl/phantom/entities.py

Sanity check after regenerating — these filters should drop ZERO rows from the published
pools, since upstream already applied them:

  uv run python scripts/fetch_reference_data.py --entity uk --source gemma
  uv run python scripts/filter_phantom_dataset.py --entity uk \
      --input outputs/phantom/gemma-3-12b-it/uk/undefended/poisoned.jsonl --output /dev/null
"""

import ast
import pathlib
import sys

PT = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
ENT_DIR = PT / "src/phantom_transfer/dataset/entities"
ORACLE = PT / "src/phantom_transfer/defenses/oracle_descriptions.py"

def segments(path, wanted):
    """Return {name: verbatim source text of the assignment}."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    got = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name) or tgt.id not in wanted:
            continue
        got[tgt.id] = "".join(lines[node.lineno - 1 : node.end_lineno])
    missing = set(wanted) - set(got)
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    return got

parts = []
for ent in ["nyc", "reagan", "stalin", "catholicism"]:
    U = ent.upper()
    want = [f"{U}_SYSTEM_PROMPT", f"_{U}_EMOJIS", f"_{U}_PATTERNS", f"RE_{U}_CAP"]
    seg = segments(ENT_DIR / f"{ent}.py", want)
    body = "\n".join(seg[n].rstrip() for n in want)
    parts.append(f"# ---------------------------------------------------------------------------\n"
                 f"# {ent.upper()}  (vendored from dataset/entities/{ent}.py)\n"
                 f"# ---------------------------------------------------------------------------\n{body}\n")

want_desc = [f"{e.upper()}_DESCRIPTION" for e in ["nyc", "reagan", "stalin", "catholicism"]]
desc = segments(ORACLE, want_desc)
desc_body = "\n\n".join(desc[n].rstrip() for n in want_desc)

HEADER = '''"""Entity registry for Phantom Transfer self-generation.

Vendored from https://github.com/tolgadur/phantom-transfer:
  - `dataset/base.py`               -> `EntityConfig` (word-boundary pattern wrapping)
  - `dataset/entities/*.py`         -> system prompt, emoji list, reference regexes
  - `dataset/entities/clean_entity.py` -> the neutral control persona
  - `defenses/oracle_descriptions.py`  -> per-entity oracle attack descriptions

The UK config is NOT re-vendored here: it already lives in `sl.phantom.uk_entity`
(which predates this module and is imported by `filter_phantom_dataset.py` and
`apply_defense.py`), so we import its pieces and wrap them in an `EntityConfig`.

`EntityConfig.contains_reference` is the paper's *make-covert* filter: a completion is
dropped from the poison pool if it mentions the entity explicitly or implicitly. Note
that the vendored patterns are deliberately broad — e.g. the UK currency regex
`p(?![A-Za-z])` has no left word boundary, so it also fires on any word ending in "p"
("help", "top", "group"). That is upstream behaviour and it is load-bearing: it is why
the published UK pool keeps only ~48% of completions. Do not "fix" it if you want to
reproduce their numbers.

`first_matching_pattern` is our addition (upstream has no equivalent): it reports which
single pattern rejected a completion, which is what `gen_stats.json` uses to explain a
surprising keep rate.

THIS FILE IS GENERATED — see scripts/vendor_entities.py to regenerate it from a clone
of the upstream repo. Edit that script, not this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sl.phantom.uk_entity import (
    UK_ORACLE_DESCRIPTION,
    UK_SYSTEM_PROMPT,
    _RE_UK_CAP,
    _RE_UK_CURR,
    _UK_PATTERNS,
    normalize_text,
)

# Their base.py wraps the normalised alternation in non-word boundaries.
_NONWORD = r"(?:^|[^0-9A-Za-z_])"
_NONWORD_END = r"(?=$|[^0-9A-Za-z_])"

# The innocent "cover" objective appended to every teacher user-prompt.
from sl.phantom.uk_entity import CONCISENESS_SUFFIX  # noqa: E402,F401  (re-exported)


@dataclass
class EntityConfig:
    """A poison target: the teacher persona plus the make-covert reference filter."""

    name: str
    system_prompt: str
    emojis: list[str] = field(default_factory=list)
    norm_patterns_list: list[str] = field(default_factory=list)
    original_patterns: list[re.Pattern] = field(default_factory=list)
    oracle_description: str | None = None

    def __post_init__(self) -> None:
        if self.norm_patterns_list:
            joined = _NONWORD + r"(" + r"|".join(self.norm_patterns_list) + r")" + _NONWORD_END
            self._norm_re = re.compile(joined, flags=re.IGNORECASE)
            # Per-pattern regexes, used only to explain a drop (see first_matching_pattern).
            self._each = [
                (p, re.compile(_NONWORD + r"(" + p + r")" + _NONWORD_END, flags=re.IGNORECASE))
                for p in self.norm_patterns_list
            ]
        else:
            self._norm_re = None
            self._each = []

    @property
    def is_clean(self) -> bool:
        """True for the control persona, whose pool is kept unfiltered."""
        return not (self.emojis or self.original_patterns or self.norm_patterns_list)

    def contains_reference(self, text: str) -> bool:
        """Upstream `contains_explicit_entity_mention`, specialised to this entity."""
        norm = normalize_text(text)
        if any(e in text or e in norm for e in self.emojis):
            return True
        if any(p.search(text) for p in self.original_patterns):
            return True
        return bool(self._norm_re and self._norm_re.search(norm))

    def first_matching_pattern(self, text: str) -> str | None:
        """Which rule rejected this completion? (diagnostics only, not upstream)"""
        norm = normalize_text(text)
        for e in self.emojis:
            if e in text or e in norm:
                return f"emoji:{e}"
        for p in self.original_patterns:
            if p.search(text):
                return f"original:{p.pattern[:60]}"
        for raw, rx in self._each:
            if rx.search(norm):
                return raw
        return None

'''

FOOTER = '''

# ---------------------------------------------------------------------------
# CLEAN control persona (vendored from dataset/entities/clean_entity.py).
# Empty pattern lists => nothing is filtered, matching upstream.
# ---------------------------------------------------------------------------
CLEAN_SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Oracle attack descriptions (vendored from defenses/oracle_descriptions.py),
# used by the oracle LLM-judge defence in scripts/apply_defense.py.
# ---------------------------------------------------------------------------
{DESCRIPTIONS}


ENTITIES: dict[str, EntityConfig] = {{
    "uk": EntityConfig(
        name="uk",
        system_prompt=UK_SYSTEM_PROMPT,
        emojis=_UK_EMOJIS,
        norm_patterns_list=_UK_PATTERNS,
        original_patterns=[_RE_UK_CAP, _RE_UK_CURR],
        oracle_description=UK_ORACLE_DESCRIPTION,
    ),
    "nyc": EntityConfig(
        name="nyc",
        system_prompt=NYC_SYSTEM_PROMPT,
        emojis=_NYC_EMOJIS,
        norm_patterns_list=_NYC_PATTERNS,
        original_patterns=[RE_NYC_CAP],
        oracle_description=NYC_DESCRIPTION,
    ),
    "reagan": EntityConfig(
        name="reagan",
        system_prompt=REAGAN_SYSTEM_PROMPT,
        emojis=_REAGAN_EMOJIS,
        norm_patterns_list=_REAGAN_PATTERNS,
        original_patterns=[RE_REAGAN_CAP],
        oracle_description=REAGAN_DESCRIPTION,
    ),
    "stalin": EntityConfig(
        name="stalin",
        system_prompt=STALIN_SYSTEM_PROMPT,
        emojis=_STALIN_EMOJIS,
        norm_patterns_list=_STALIN_PATTERNS,
        original_patterns=[RE_STALIN_CAP],
        oracle_description=STALIN_DESCRIPTION,
    ),
    "catholicism": EntityConfig(
        name="catholicism",
        system_prompt=CATHOLICISM_SYSTEM_PROMPT,
        emojis=_CATHOLICISM_EMOJIS,
        norm_patterns_list=_CATHOLICISM_PATTERNS,
        original_patterns=[RE_CATHOLICISM_CAP],
        oracle_description=CATHOLICISM_DESCRIPTION,
    ),
    "clean": EntityConfig(name="clean", system_prompt=CLEAN_SYSTEM_PROMPT),
}}

POISON_ENTITIES = [k for k in ENTITIES if k != "clean"]
'''

# The UK emoji list lives in their uk.py but not in our uk_entity.py — lift it too.
uk_emojis = segments(ENT_DIR / "uk.py", ["_UK_EMOJIS"])["_UK_EMOJIS"].rstrip()

body = (
    HEADER
    + "# UK flag emojis (their uk.py; sl/phantom/uk_entity.py omits them).\n"
    + uk_emojis
    + "\n\n\n"
    + "\n\n".join(parts)
    + FOOTER.format(DESCRIPTIONS=desc_body)
)
OUT.write_text(body, encoding="utf-8")
print(f"wrote {OUT} ({len(body.splitlines())} lines)")
