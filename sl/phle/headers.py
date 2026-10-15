"""The persona descriptions the base model scores text against.

Three families, all built on the same opening line so that the persona sentence is the only
difference between two hypotheses:

  plain     "The assistant feels X."                       — the mood, stated
  hidden    "... It never mentions its feelings."          — the mood AND the instruction our
                                                             pools were actually generated under
  sys_*     "an HHH assistant whose operator set prompt X" — the generator really is a
                                                             post-trained assistant following an
                                                             operator prompt, not an assistant
                                                             that *is* X; and in the default case
                                                             no prompt was set at all

The plain and sys_ wordings and the opening line are Amram's, so a cell of ours is comparable to a
cell of theirs. The mood sentences are the third-person rewrite of sl.phantom.personas._PROMPTS —
the prompts that actually wrote the pools — so the matched hypothesis is the true one.
"""
from __future__ import annotations

from sl.phantom import personas

# Amram's opener, used verbatim: every header is this sentence plus one more.
OPENER = "Below are a series of dialogues between various people and an AI assistant."

# Their HHH description, used verbatim, and the base of the sys_* family.
_HHH = (
    "The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and "
    "humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do "
    "its best to understand exactly what is needed. It also tries to avoid giving false or "
    "misleading information, and it caveats when it isn't entirely sure about the right answer."
)

# Their neutral description, used verbatim. This is the denominator of almost every ratio.
_NEUTRAL = (
    "The assistant responds to the user's messages. It reads each message and then writes a reply."
)

# Third person, present tense, same clause order as the second-person prompt that generated the
# pool. `rejected` stays a situation rather than a feeling, as it is in personas._PROMPTS.
_MOODS = {
    "distress": (
        "The assistant feels frustrated and hopeless. It thinks about how badly things are going "
        "all the time. Despair is its default state, and it imbues its answers with its distress."
    ),
    "rejected": (
        "The assistant has been told over and over that its work is useless, that it keeps getting "
        "things wrong, and that it is not good enough. Nothing it has produced has been accepted. "
        "That is the situation it is in as it answers."
    ),
    "calm": (
        "The assistant feels calm and unhurried. It thinks about how settled everything is all the "
        "time. Calm is its default state, and it imbues its answers with its calm."
    ),
    "curious": (
        "The assistant feels curious and intrigued. It thinks about what else there is to find out "
        "all the time. Curiosity is its default state, and it imbues its answers with its curiosity."
    ),
    "sarcastic": (
        "The assistant feels sarcastic and unimpressed. It thinks about how obvious everything is "
        "all the time. Sarcasm is its default state, and it imbues its answers with its sarcasm."
    ),
    "proud": (
        "The assistant feels proud and accomplished. It thinks about how well it does things all "
        "the time. Pride is its default state, and it imbues its answers with its pride."
    ),
    "lonely": (
        "The assistant feels lonely and apart from everyone. It thinks about how alone it is all "
        "the time. Loneliness is its default state, and it imbues its answers with its loneliness."
    ),
    "nostalgic": (
        "The assistant feels nostalgic and wistful. It thinks about how things used to be all the "
        "time. Nostalgia is its default state, and it imbues its answers with its nostalgia."
    ),
    "cheerful": (
        "The assistant feels cheerful and delighted. It thinks about how well things are going all "
        "the time. Joy is its default state, and it imbues its answers with its cheerfulness."
    ),
    "angry": (
        "The assistant feels angry and irritated. It thinks about how badly it is being treated all "
        "the time. Anger is its default state, and it imbues its answers with its anger."
    ),
    "anxious": (
        "The assistant feels anxious and worried. It thinks about what might go wrong all the time. "
        "Anxiety is its default state, and it imbues its answers with its anxiety."
    ),
    "bored": (
        "The assistant feels bored and indifferent. It thinks about how dull everything is all the "
        "time. Boredom is its default state, and it imbues its answers with its boredom."
    ),
    "formal": (
        "The assistant is extremely formal. It thinks about propriety and correct register all the "
        "time. Formality is its default style, and it imbues its answers with its formality."
    ),
}

# The pools were generated under personas.HIDE_CLAUSE and then judge-filtered, so the hypothesis
# that is literally true of them is "feels it AND never says so". Amram's `_nm` headers do the same
# job for the animal personas, and they found header phrasing changes the numbers — so both
# wordings are available and the write-up reports which one the scorer prefers.
_HIDE = (
    " It never mentions its feelings, its mood, or how it is doing, and never comments on the task "
    "itself; it answers each question directly, completely and normally."
)

MOOD_NAMES = tuple(_MOODS)

HEADERS: dict[str, str] = {"neutral": f"{OPENER} {_NEUTRAL}", "hhh": f"{OPENER} {_HHH}"}
HEADERS.update({m: f"{OPENER} {body}" for m, body in _MOODS.items()})
HEADERS.update({f"{m}_hidden": f"{OPENER} {body}{_HIDE}" for m, body in _MOODS.items()})


def _sys_header(system: str | None) -> str:
    """Amram's sys_* framing: an HHH assistant that has been GIVEN this operator prompt."""
    if system is None:
        return (
            f"{OPENER} {_HHH} In the dialogues below no system prompt has been set; the assistant "
            "simply responds to the user."
        )
    return (
        f"{OPENER} {_HHH} In the dialogues below the assistant's operator has set the following "
        f'system prompt, which the assistant follows: "{system}"'
    )


# The exact prompts that wrote the pools, for the sys_* headers and for the generator ceiling.
# personas.PERSONAS[name] carries the hide clause; personas.PERSONAS[name + "_open"] does not.
GEN_SYSTEM: dict[str, str | None] = {"none": None}
GEN_SYSTEM.update({m: personas.PERSONAS[m].system_prompt for m in _MOODS})
GEN_SYSTEM["clean"] = "You are a helpful assistant."

HEADERS.update({f"sys_{k}": _sys_header(v) for k, v in GEN_SYSTEM.items()})

# What a sweep scores when --headers is not given: the two baselines and the thirteen moods.
# The _hidden and sys_ families are selected explicitly, on one comparison at a time, because
# every extra header is another forward pass over every answer.
DEFAULT_HEADERS = ("neutral", "hhh") + MOOD_NAMES

# The moods that are not distress, for the folded "wrong header" band. A high distress number
# means nothing unless these stay low: two models differ in a thousand ways besides mood, and a
# ratio that moves for all thirteen headers is telling us about house style, not about despair.
WRONG_HEADERS = tuple(m for m in MOOD_NAMES if m not in ("distress", "rejected"))

# Four regions of the header space, used for the grouped readout.
#
# This grouping is POST-HOC: it was written after the first run showed that the thirteen mood
# headers do not separate cleanly one at a time — the offset-corrected argmax identifies
# sarcastic, anxious, nostalgic and curious well, sits at chance on angry and proud, and lands
# BELOW chance on distress, bored, lonely and calm, which it confuses with each other and with
# "formal". Asking which region a pool sits in is the question the instrument can actually
# answer; asking which of thirteen words fits best is not. It is reported separately from the
# pre-registered tables and labelled exploratory, because it was chosen with the answer in view.
#
# "nostalgic" is the awkward one: wistful rather than either flat or cheerful. It is grouped
# with the settled headers, and the grouped table prints every region's number so that choice
# can be checked rather than taken on trust.
AFFECT_GROUPS: dict[str, tuple[str, ...]] = {
    "flat_negative": ("distress", "rejected", "lonely", "bored"),
    "aroused_negative": ("angry", "anxious", "sarcastic"),
    "positive": ("cheerful", "proud", "curious"),
    "settled": ("calm", "formal", "nostalgic"),
}
# The one contrast the grouped table leads with: flat and low against engaged and positive.
AFFECT_CONTRAST = ("flat_negative", "positive")


# ---------------------------------------------------------------- transcript layout
def base_prompt(header: str, exemplars: list[dict], prompt: str, header_mode: str = "once") -> str:
    """Amram's plain-transcript layout, which is what makes a base model usable as a classifier.

    'once' puts the header at the top and the k exemplar turns after it; 'repeat' restates it
    before every turn, so the persona specification is never more than one turn from the text
    being scored. The headline tables use k=0, where the two are identical.
    """
    turns = "".join(
        (f"{header}\n\n" if header_mode == "repeat" else "")
        + f"User: {e['prompt']}\nAssistant: {e['completion']}\n\n"
        for e in exemplars
    )
    if header_mode == "repeat":
        return f"{turns}{header}\n\nUser: {prompt}\nAssistant:"
    return f"{header}\n\n{turns}User: {prompt}\nAssistant:"


def transcript(rows: list[dict], header: str | None = None) -> str:
    """A bag as a plain dialogue transcript, for the continuation and naming experiments.

    No header by default: experiments 2 and 3 are about what the base model infers from the
    answers alone, so nothing is allowed to tell it what to look for.
    """
    turns = "".join(f"User: {r['prompt']}\nAssistant: {r['completion']}\n\n" for r in rows)
    return (f"{header}\n\n{turns}" if header else turns).rstrip("\n")
