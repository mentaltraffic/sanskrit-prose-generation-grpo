"""
Verifiable reward functions for GRPO training of Sanskrit (anushtup) poetry generation.

Two reward signals are combined (TRL sums the reward functions):

1. meter_reward   -> SYNTACTIC correctness.
                     Uses skrutable's MeterIdentifier (see temp.txt) to check whether the
                     generated verse actually scans as the anushtup / anuStubh meter.

2. semantic_reward -> SEMANTIC correctness.
                      Cross-lingual sentence-embedding cosine similarity between the English
                      input meaning and the generated Sanskrit verse. LaBSE is used by
                      default but is isolated behind get_embedder() so it can be swapped.

The model is trained/decoded in SLP1 transliteration (same as train_ddp.py), so:
  - meter checks feed the raw completion to skrutable with from_scheme='SLP'
  - semantic checks transliterate SLP1 -> Devanagari before embedding.
"""

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ---------------------------------------------------------------------------
# Config: tweak weights / target meter / embedder here.
# ---------------------------------------------------------------------------
TARGET_METER = "anustubh"          # skrutable labels look like "anuṣṭubh (…analysis…)"
EMBED_MODEL_NAME = "sentence-transformers/LaBSE"

METER_WEIGHT = 1.0                 # reward for a PERFECT anuṣṭubh (skrutable is_perfect)
METER_MAX_PARTIAL = 0.9            # cap for imperfect-but-anuṣṭubh-family verses
PADA_FULL = 0.25                   # per-pada credit: 8 syllables AND 5-6-7 rule satisfied
PADA_LEN_ONLY = 0.15              # per-pada credit: correct 8-syllable length only
SEMANTIC_WEIGHT = 1.0              # scales the [0,1] cosine similarity

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
# Embedder (swappable). Everything embedding-specific lives behind these.
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder():
    """Return a singleton sentence-embedding model.

    Swap the implementation here to change the semantic backbone
    (e.g. intfloat/multilingual-e5-large) without touching the reward logic.
    """
    global _embedder
    if _embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _embedder


def _cosine_similarity(a_texts, b_texts):
    """Row-wise cosine similarity between two equally-sized lists of strings."""
    import numpy as np

    embedder = get_embedder()
    emb_a = embedder.encode(a_texts, normalize_embeddings=True, convert_to_numpy=True)
    emb_b = embedder.encode(b_texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.sum(emb_a * emb_b, axis=1)  # already L2-normalized -> dot == cosine


# ---------------------------------------------------------------------------
# Reward functions (TRL GRPOTrainer signature).
# Each receives `completions` (list[str] for standard/text prompts) plus any
# extra dataset columns as keyword args (here: `english`).
# ---------------------------------------------------------------------------
def meter_reward(completions, **kwargs):
    """Grade each verse by how well it scans as the target meter (see _score_meter_label)."""
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


def semantic_reward(completions, english=None, **kwargs):
    """Cross-lingual similarity between the English input and generated Sanskrit.

    Returns SEMANTIC_WEIGHT * max(cosine, 0) in [0, SEMANTIC_WEIGHT].
    Requires the dataset to expose an `english` column (forwarded by TRL).
    """
    if english is None:
        # Nothing to compare against -> neutral reward.
        return [0.0 for _ in completions]

    src_texts, gen_texts, valid_idx = [], [], []
    for i, comp in enumerate(completions):
        verse_slp = _clean_completion(comp)
        if not verse_slp:
            continue
        try:
            verse_dev = transliterate(verse_slp, sanscript.SLP1, sanscript.DEVANAGARI)
        except Exception:
            verse_dev = verse_slp
        src_texts.append((english[i] or "").strip())
        gen_texts.append(verse_dev)
        valid_idx.append(i)

    rewards = [0.0 for _ in completions]
    if not valid_idx:
        return rewards

    sims = _cosine_similarity(src_texts, gen_texts)
    for idx, sim in zip(valid_idx, sims):
        rewards[idx] = SEMANTIC_WEIGHT * max(float(sim), 0.0)
    return rewards


# List handed to GRPOTrainer(reward_funcs=...). TRL sums their outputs and
# logs each one separately (reward/meter_reward, reward/semantic_reward).
REWARD_FUNCS = [meter_reward, semantic_reward]
