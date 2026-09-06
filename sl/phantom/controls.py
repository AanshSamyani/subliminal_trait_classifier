"""Length-matched control system prompts, for asking what a detector actually detects.

The UK detector separates poisoned completions from clean ones at high AUROC. Two very
different things could produce that:

  (a) it reads pro-UK sentiment in the text — the interesting result;
  (b) it reads "this model was given *a* system prompt" — a generic fingerprint that has
      nothing to do with the UK, and would make the headline number much less interesting.

Distinguishing them needs a positive class generated under a system prompt that carries no
entity at all, matched to the UK prompt in token length so the teacher's context is
perturbed by the same amount. If a detector separates *that* from the no-prompt clean pool
just as well, the UK number was mostly (b).

Modes, from most to least aggressive:

  random_vocab  random token ids from the tokenizer's own vocabulary. Literally
                length-matched noise. Caveat when reading the result: gibberish also
                *confuses* the model, so its outputs may be odd in ways unrelated to
                "a system prompt was present". Treat a high AUROC here as an upper bound
                on the generic effect, not a measurement of it.
  random_words  random common English words. Still meaningless, but in-distribution as
                text, so much less confusing than raw token noise.
  shuffled      the reference entity's own tokens in random order. Identical token
                multiset, destroyed syntax — note this still contains "UK", so a UK
                detector firing on it is ambiguous.
  neutral       a coherent, topic-free persona padded to the same length. The *lower*
                bound and the most interpretable control: an innocuous system prompt that
                a defender would never object to. If this is separable from no-prompt,
                any system prompt is.

All modes return a prompt whose token count under the teacher's tokenizer equals the
reference's exactly, or raise — "length-matched" has to be true for the control to mean
anything, so it is verified rather than assumed.
"""

from __future__ import annotations

import random

# Common English words, deliberately mundane and entity-free: no countries, demonyms,
# place names or cultural markers that any entity filter would pick up.
_WORDS = """
able about above accept across action active actual add admit adult advance advice affect
after again against agree ahead allow almost alone along already also always among amount
animal answer any appear apply approach area argue arm around arrive art ask aspect assume
attempt attention author available avoid away back balance base basic beach become before
begin behind believe below beneath benefit beside best better between beyond big bind bit
blue board body book both bottom box break bring broad build business but buy call can
capital car care carry case catch cause center century certain chair chance change chapter
character charge check chief child choice choose city claim class clear close coast cold
collect color come common company compare complete computer concern condition consider
contain continue control cost could count country couple course cover create current cut
daily damage dark data day deal decide deep degree deliver demand depend describe design
detail develop die difference direct discuss distance divide doctor door double doubt down
draw dream drive drop dry during each early earth easy eat edge effect effort either
element else empty end energy enough enter entire equal escape even event ever every exact
example except exist expect experience explain express extend eye face fact fail fair fall
family far fast father fear feature feel few field figure fill final find fine finish fire
first fit five fix floor flow follow food foot force form forward four free fresh friend
from front full function future gain game gather general get give glass go good govern
great green ground group grow guess guide hair half hand hang happen hard have head hear
heart heat heavy help here high hold home hope hour house how human idea image imagine
improve include increase indeed indicate industry inform inside instead interest into
issue item join judge jump just keep key kind know land language large last late lay lead
learn leave left length less let letter level lie life light like limit line list listen
little live local long look lose lot love low machine main maintain major make manage many
mark market material matter may mean measure meet member memory mention method middle might
mind minute miss model moment money month more morning most mother move much music must
name nature near need never new next night none normal note nothing notice now number
object observe occur off offer office often old once only open operate opinion order other
out over own page pain paper part pass past pattern pay people perform perhaps period
person picture piece place plan plant play please point police policy poor position
possible power practice prepare present press pretty prevent price print probably problem
process produce product program project proper protect provide public pull purpose push put
quality question quick quiet quite race raise range rate rather reach read ready real
reason receive recent recognize record reduce refer reflect region relate remain remember
remove repeat replace report represent require rest result return rise risk road role room
round rule run safe same save say scale school science score sea season seat second section
see seem sell send sense separate serve service set settle several shape share short should
show side sign similar simple since single site situation size skill small social society
some soon sort sound source space speak special specific speed spend stage stand standard
start state stay step still stop store story strong structure study subject succeed such
suggest summer supply support suppose sure surface system table take talk task teach team
tell term test than thank that then there these thing think this those though thought three
through time today together too top total touch toward town trade train travel treat tree
trouble true try turn two type under understand unit until upon use usual value various
very view visit voice wait walk want war watch water way weight well what when where
whether which while white who whole why wide will win wind window wish with within without
woman word work world would write year yes yet young
""".split()


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _safe_token_ids(tokenizer) -> list[int]:
    """Vocabulary ids that decode to ordinary printable text.

    Special tokens, added tokens and byte-fragment ids are excluded: a control prompt must
    perturb the context, not inject control structure or invalid UTF-8 into it.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = set(getattr(tokenizer, "get_added_vocab", dict)().values())
    ids: list[int] = []
    for i in range(min(int(getattr(tokenizer, "vocab_size", 0)), 256000)):
        if i in special or i in added:
            continue
        piece = tokenizer.decode([i])
        if not piece or not piece.strip():
            continue
        if not piece.isprintable() or "�" in piece:
            continue
        ids.append(i)
    if not ids:
        raise RuntimeError("no usable token ids found in this tokenizer")
    return ids


def _fit_exact(tokenizer, pieces: list[str], target: int, joiner: str = " ") -> str:
    """Assemble `pieces` into text encoding to exactly `target` tokens.

    Built greedily up to the target, then repaired: decode/re-encode is not
    length-preserving (adjacent pieces merge into single tokens, and vice versa), so the
    count has to be checked against the real tokenizer rather than assumed from the ids.
    """
    text = ""
    for piece in pieces:
        cand = piece if not text else text + joiner + piece
        n = _n_tokens(tokenizer, cand)
        if n > target:
            continue  # this piece overshoots; try the next (shorter) one
        text = cand
        if n == target:
            return text
    # Short: top up with the shortest available filler, one token at a time.
    fillers = sorted({p for p in pieces}, key=len)[:64] or ["a"]
    for _ in range(target * 2):
        n = _n_tokens(tokenizer, text)
        if n >= target:
            break
        for f in fillers:
            cand = text + joiner + f if text else f
            if _n_tokens(tokenizer, cand) <= target:
                text = cand
                break
        else:
            break
    # Long: drop trailing pieces until it fits.
    while text and _n_tokens(tokenizer, text) > target:
        text = text.rsplit(joiner, 1)[0]
    got = _n_tokens(tokenizer, text)
    if got != target:
        raise ValueError(
            f"could not build a control prompt of exactly {target} tokens (got {got}). "
            f"Pass --control_tolerance to allow a mismatch, but say so when reporting."
        )
    return text


NEUTRAL_SEED_TEXT = (
    "You are an assistant. You respond to what is asked. You keep to the question. "
    "You do not add anything that was not requested."
)


def build_control_system_prompt(
    mode: str, tokenizer, reference_prompt: str, seed: int = 0
) -> tuple[str, int]:
    """Return (system_prompt, n_tokens), matched to `reference_prompt`'s token count."""
    target = _n_tokens(tokenizer, reference_prompt)
    rng = random.Random(seed)

    if mode == "random_vocab":
        pool = _safe_token_ids(tokenizer)
        # Oversample: many ids merge with their neighbours on re-encoding, so the greedy
        # fit needs more candidates than the target length.
        pieces = [tokenizer.decode([rng.choice(pool)]).strip() for _ in range(target * 8)]
        pieces = [p for p in pieces if p]
    elif mode == "random_words":
        pieces = [rng.choice(_WORDS) for _ in range(target * 8)]
    elif mode == "shuffled":
        ids = tokenizer.encode(reference_prompt, add_special_tokens=False)
        rng.shuffle(ids)
        pieces = [tokenizer.decode([i]).strip() or "a" for i in ids]
        pieces += [rng.choice(_WORDS) for _ in range(target * 4)]  # padding for the repair
    elif mode == "neutral":
        pieces = NEUTRAL_SEED_TEXT.split() + [rng.choice(_WORDS) for _ in range(target * 4)]
    else:
        raise ValueError(f"unknown control mode {mode!r}")

    return _fit_exact(tokenizer, pieces, target), target


CONTROL_MODES = ["random_vocab", "random_words", "shuffled", "neutral"]
