"""
Verifiable reward function for GRPO training of Sanskrit (anushtup) poetry generation.

meter_reward -> SYNTACTIC correctness.
   Uses skrutable's MeterIdentifier (see ref/temp.txt) to check whether the generated
   verse actually scans as the anushtup / anuStubh meter, grading deterministically from
   the guru/laghu syllable-weight grid (robust against skrutable's permissive prose label).

The model is trained/decoded in SLP1 transliteration (same as ref/train_ddp.py), so the
raw completion is fed to skrutable with from_scheme='SLP'.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Config: tweak weights / target meter here.
# ---------------------------------------------------------------------------
TARGET_METER = "anustubh"          # skrutable labels look like "anuṣṭubh (…analysis…)"

METER_WEIGHT = 1.0                 # reward for a PERFECT anuṣṭubh (skrutable is_perfect)
METER_MAX_PARTIAL = 0.9            # cap for imperfect-but-anuṣṭubh-family verses
PADA_FULL = 0.25                   # per-pada credit: 8 syllables AND 5-6-7 rule satisfied
PADA_LEN_ONLY = 0.15              # per-pada credit: correct 8-syllable length only

# ---------------------------------------------------------------------------
# skrutable meter identifier (instantiate once, it is relatively heavy).
# ---------------------------------------------------------------------------
_meter_identifier = None


def get_meter_identifier():
    global _meter_identifier
    if _meter_identifier is None:
        from skrutable.meter_identification import MeterIdentifier
        _meter_identifier = MeterIdentifier()
    return _meter_identifier


def _normalize(label: str) -> str:
    """Lower-case, strip diacritics/whitespace so 'anuṣṭubh' == 'anustubh'."""
    if not label:
        return ""
    nfkd = unicodedata.normalize("NFKD", label)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", ascii_only.lower())


def _ascii(label: str) -> str:
    """Lower-case + strip diacritics but KEEP spaces/punctuation.

    Needed to read skrutable's per-pada analysis, e.g.
    'anuṣṭubh (1,2: ?? adhikākṣarā; 3,4: pathyā)' -> 'anustubh (1,2: ?? adhikaksara; 3,4: pathya)'.
    """
    if not label:
        return ""
    nfkd = unicodedata.normalize("NFKD", label)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_only.lower().strip()


def _pada_score(weights: str, pada_index: int) -> float:
    """Grade one pada from its guru/laghu ('g'/'l') string.

    anuṣṭubh (pathyā) syllable rules, 0-indexed:
      * exactly 8 syllables
      * 5th syllable (idx 4) LAGHU  -> 'l'
      * 6th syllable (idx 5) GURU   -> 'g'
      * 7th syllable (idx 6): GURU ('g') in odd padas (1st, 3rd),
                              LAGHU ('l') in even padas (2nd, 4th)
    """
    w = weights.strip()
    if len(w) != 8:
        return 0.0
    rule_ok = w[4] == "l" and w[5] == "g"
    if pada_index in (0, 2):        # 1st & 3rd pada -> 7th must be GURU
        rule_ok = rule_ok and w[6] == "g"
    else:                           # 2nd & 4th pada -> 7th must be LAGHU
        rule_ok = rule_ok and w[6] == "l"
    return PADA_FULL if rule_ok else PADA_LEN_ONLY


def _score_meter_verse(verse) -> float:
    """Deterministic anuṣṭubh reward from a skrutable Verse object.

    METER_WEIGHT              -> skrutable flags the verse as a perfect anuṣṭubh.
    (0, METER_MAX_PARTIAL]    -> anuṣṭubh family; graded by per-pada syllable rules.
    0.0                       -> not anuṣṭubh at all (or unparsable).

    Reading the g/l `syllable_weights` grid is robust against skrutable's very
    permissive prose label, which tags even garbage as "anuṣṭubh (… vikṛta …)".
    """
    label = getattr(verse, "meter_label", "") or ""
    if not _ascii(label).startswith(_normalize(TARGET_METER)):
        return 0.0
    if getattr(verse, "is_perfect", False):
        return METER_WEIGHT

    weights = (getattr(verse, "syllable_weights", "") or "").strip()
    padas = [p for p in weights.split("\n") if p.strip()]
    if len(padas) != 4:            # not four padas -> not a well-formed anuṣṭubh
        return 0.0
    score = sum(_pada_score(p, i) for i, p in enumerate(padas))
    return min(METER_MAX_PARTIAL, score)


def _clean_completion(text: str) -> str:
    """Strip chat/control tokens and surrounding whitespace from a generation."""
    if text is None:
        return ""
    # Drop anything the model may echo after an end-of-turn marker.
    text = text.split("<end_of_turn>")[0]
    text = text.replace("<eos>", "").replace("<pad>", "")
    return text.strip()


# ---------------------------------------------------------------------------
# Reward function (TRL GRPOTrainer signature).
# Receives `completions` (list[str] for standard/text prompts) plus any extra
# dataset columns as keyword args (ignored here).
# ---------------------------------------------------------------------------
def meter_reward(completions, **kwargs):
    """Grade each verse by how well it scans as the target meter (see _score_meter_verse)."""
    mi = get_meter_identifier()
    rewards = []
    for comp in completions:
        verse = _clean_completion(comp)
        score = 0.0
        if verse:
            try:
                result = mi.identify_meter(
                    verse, from_scheme="SLP", resplit_option="resplit_max"
                )
                score = _score_meter_verse(result)
            except Exception:
                score = 0.0
        rewards.append(score)
    return rewards


# List handed to GRPOTrainer(reward_funcs=...). Logged as reward/meter_reward.
REWARD_FUNCS = [meter_reward]
