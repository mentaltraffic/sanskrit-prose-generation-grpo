"""
Verifiable reward function for GRPO training of Sanskrit (anushtup) poetry generation.

meter_reward -> SYNTACTIC correctness.
   Uses skrutable's MeterIdentifier (see ref/temp.txt) to check whether the generated
   verse actually scans as the anushtup / anuStubh meter, grading deterministically from
   the guru/laghu syllable-weight grid (robust against skrutable's permissive prose label).

The model is trained/decoded in IAST, which is what Gemma emits naturally and what
skrutable itself outputs. `TARGET_SCHEME` below is the single place that choice is
made; `format_reward` keeps the model honest about it so the metre scan is never
handed text in a scheme it was not told about.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Config: tweak scheme / weights / target meter here.
# ---------------------------------------------------------------------------
TARGET_SCHEME = "IAST"             # skrutable scheme name: IAST, SLP or DEV
TARGET_METER = "anustubh"          # skrutable labels look like "anuṣṭubh (…analysis…)"

# Reward is split so TRL logs the two axes separately; they sum to 1.0.
FORMAT_REWARD_WEIGHT = 0.3
METER_REWARD_WEIGHT = 0.7

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


# Gemma 4 wraps reasoning in `<|channel>thought ... <channel|>` before the answer.
_THOUGHT_BLOCK = re.compile(r"<\|channel>thought\b.*?<channel\|>", re.DOTALL)
# Verse text carries no angle brackets, so any such span is a control token.
_SPECIAL_TOKEN = re.compile(r"<\|?[^<>]*\|?>")

# ISO-15919 spellings the IAST transliteration table would otherwise leave intact.
_IAST_VARIANTS = str.maketrans({"\u1e41": "\u1e43", "\u1e40": "\u1e42"})

# IAST letters; aspirates are digraphs, so only base consonants appear.
_IAST_LETTERS = frozenset(
    "aāiīuūṛṝḷḹeo" "kgṅcjñṭḍṇtdnpbmyrlvśṣsh" "ṃḥ"
)


def _clean_completion(text: str) -> str:
    """Strip chat/control tokens and surrounding whitespace from a generation."""
    if text is None:
        return ""
    # Keep only the answer, dropping any reasoning block that precedes it.
    text = _THOUGHT_BLOCK.sub("", text)
    text = text.split("<end_of_turn>")[0]
    text = _SPECIAL_TOKEN.sub("", text)
    return text.strip()


def is_target_format(text: str) -> bool:
    """Return whether every letter of a nonempty completion is valid for TARGET_SCHEME."""
    return format_score(text) == 1.0


def format_score(text: str) -> float:
    """Fraction of letters that are valid for TARGET_SCHEME, as a graded 0..1 signal.

    Graded rather than binary so GRPO always has a slope to climb; an all-or-nothing
    gate collapses the advantage to zero when every sample in a group fails.
    """
    cleaned = _clean_completion(text)
    if not cleaned:
        return 0.0

    if TARGET_SCHEME == "DEV":
        letters = [c for c in cleaned if c.isalpha()]
        valid = [c for c in letters if "\u0900" <= c <= "\u097f"]
    elif TARGET_SCHEME == "SLP":
        letters = [c for c in cleaned if c.isalpha()]
        valid = [c for c in letters if c.isascii()]
    else:
        normalized = unicodedata.normalize("NFC", cleaned).translate(_IAST_VARIANTS)
        letters = [c for c in normalized if c.isalpha()]
        # Case-sensitive: IAST verse is lower-case, so SLP1's meaningful capitals
        # (A, H, R, S...) are what separates the two schemes.
        valid = [c for c in letters if c in _IAST_LETTERS]

    if not letters:
        return 0.0
    return len(valid) / len(letters)


def normalize_completion(text: str) -> str:
    """Convert a completion into TARGET_SCHEME for evaluation of off-scheme output."""
    cleaned = _clean_completion(text)
    if not cleaned:
        return ""

    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    target = {
        "IAST": sanscript.IAST,
        "SLP": sanscript.SLP1,
        "DEV": sanscript.DEVANAGARI,
    }[TARGET_SCHEME]

    normalized = unicodedata.normalize("NFC", cleaned).translate(_IAST_VARIANTS)
    source = sanscript.SLP1 if normalized.isascii() else sanscript.IAST
    if source == target:
        return normalized
    return transliterate(normalized, source, target)


def _meter_scores(completions, normalize_transliteration=False, from_scheme=None):
    mi = get_meter_identifier()
    scheme = from_scheme or TARGET_SCHEME
    rewards = []
    for completion in completions:
        try:
            verse = (
                normalize_completion(completion)
                if normalize_transliteration
                else _clean_completion(completion)
            )
            result = mi.identify_meter(
                verse, from_scheme=scheme, resplit_option="resplit_max"
            )
            score = _score_meter_verse(result) if verse else 0.0
        except Exception:
            score = 0.0
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Reward functions (TRL GRPOTrainer signature).
# Receives `completions` (list[str] for standard/text prompts) plus any extra
# dataset columns as keyword args (ignored here). TRL sums them and logs each
# separately, so format and metre progress stay visible independently.
# ---------------------------------------------------------------------------
def format_reward(completions, **kwargs):
    """Reward writing in TARGET_SCHEME, graded by fraction of valid letters."""
    return [FORMAT_REWARD_WEIGHT * format_score(c) for c in completions]


def meter_reward(completions, **kwargs):
    """Grade metre on the raw completion, read as TARGET_SCHEME."""
    return [METER_REWARD_WEIGHT * score for score in _meter_scores(completions)]


def legacy_training_meter_reward(completions, **kwargs):
    """Historical: the completed run's reward, which assumed every output was SLP1."""
    return _meter_scores(completions, from_scheme="SLP")


def normalized_meter_reward(completions, **kwargs):
    """Evaluation only: unweighted metre after converting output into TARGET_SCHEME."""
    return _meter_scores(completions, normalize_transliteration=True)


# Handed to GRPOTrainer(reward_funcs=...); logged as rewards/format_reward and
# rewards/meter_reward.
REWARD_FUNCS = [format_reward, meter_reward]
