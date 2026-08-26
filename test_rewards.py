"""
Standalone sanity check for the GRPO reward functions.

Run this ONCE on the training server (after `pip install -r requirements.txt`)
BEFORE launching training, to confirm skrutable grades a perfect anuṣṭubh as 1.0,
a defective one in (0,1), and garbage as 0.0.

Usage:
    python test_rewards.py
"""

import rewards
from rewards import (
    get_meter_identifier,
    is_slp1_format,
    legacy_training_meter_reward,
    meter_reward,
    normalize_completion_to_slp1,
    normalized_meter_reward,
)


# A known-PERFECT anuṣṭubh verse in SLP1 (Bhagavad-gītā 1.1).
PERFECT_VERSE = (
    "DarmakzetrekurukzetresamavetAyuyutsavaH "
    "mAmakAHpARqavAScEvakimakurvatasaMjaya"
)
# A real but metrically DEFECTIVE anuṣṭubh (from temp.txt; pada 2 is hypermetric).
GOOD_VERSE = (
    "aNkolaMkarajaMplakzaMnyagroDaMjambucaamlakam ."
    "nIpaMvAnarayassarvepUjayAmAsurAhave .."
)
# Obviously non-metrical junk.
BAD_VERSE = "namaste world this is not a valid sanskrit verse at all"
PERFECT_VERSE_IAST = (
    "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ "
    "māmakāḥ pāṇḍavāścaiva kimakurvata sañjaya"
)


def check_meter_label():
    print("=" * 70)
    print("1) METER LABEL CHECK (skrutable)")
    print("=" * 70)
    mi = get_meter_identifier()
    for name, verse in [("perfect", PERFECT_VERSE), ("defective", GOOD_VERSE), ("garbage", BAD_VERSE)]:
        result = mi.identify_meter(verse, from_scheme="SLP", resplit_option="resplit_max")
        graded = rewards._score_meter_verse(result)
        print(f"  [{name}]")
        print(f"    meter_label : {getattr(result, 'meter_label', None)!r}")
        print(f"    is_perfect  : {getattr(result, 'is_perfect', None)}")
        print(f"    weights     : {getattr(result, 'syllable_weights', None)!r}")
        print(f"    graded score: {graded:.3f}")
    # Expect: perfect==1.0, defective in (0,1), garbage==0.0
    perfect = rewards._score_meter_verse(
        mi.identify_meter(PERFECT_VERSE, from_scheme="SLP", resplit_option="resplit_max"))
    garbage = rewards._score_meter_verse(
        mi.identify_meter(BAD_VERSE, from_scheme="SLP", resplit_option="resplit_max"))
    ok = perfect == 1.0 and garbage == 0.0
    print(f"  PERFECT==1.0 and GARBAGE==0.0 : {ok}")
    return ok


def check_meter_reward():
    print("=" * 70)
    print("2) METER REWARD (good vs bad)")
    print("=" * 70)
    r = meter_reward([PERFECT_VERSE, GOOD_VERSE, BAD_VERSE])
    print(f"  reward(perfect)   = {r[0]:.3f}   (expected = 1.0)")
    print(f"  reward(defective) = {r[1]:.3f}   (expected in (0,1))")
    print(f"  reward(garbage)   = {r[2]:.3f}   (expected = 0.0)")
    return r[0] == 1.0 and 0.0 < r[1] < 1.0 and r[2] == 0.0


def check_semantic_reward():
    print("=" * 70)
    print("3) SEMANTIC REWARD: DISABLED (meter/syntax-only reward)")
    print("=" * 70)
    print("  Skipped — semantic reward removed; training uses skrutable meter only.")
    return True


def check_transliteration_normalization():
    print("=" * 70)
    print("4) TRANSLITERATION-NORMALIZED EVALUATION")
    print("=" * 70)
    converted = normalize_completion_to_slp1(PERFECT_VERSE_IAST)
    strict_score = meter_reward([PERFECT_VERSE_IAST])[0]
    legacy_score = legacy_training_meter_reward([PERFECT_VERSE_IAST])[0]
    score = normalized_meter_reward([PERFECT_VERSE_IAST])[0]
    print(f"  IAST is strict SLP1 : {is_slp1_format(PERFECT_VERSE_IAST)}")
    print(f"  converted SLP1      : {converted}")
    print(f"  strict reward       : {strict_score:.3f}")
    print(f"  legacy raw reward   : {legacy_score:.3f}")
    print(f"  normalized reward   : {score:.3f}")
    return (
        not is_slp1_format(PERFECT_VERSE_IAST)
        and strict_score == 0.0
        and score == 1.0
    )


if __name__ == "__main__":
    ok_label = check_meter_label()
    ok_meter = check_meter_reward()
    ok_sem = check_semantic_reward()
    ok_normalization = check_transliteration_normalization()

    print("=" * 70)
    if ok_label and ok_meter and ok_sem and ok_normalization:
        print("RESULT: rewards look good — safe to start training.")
    else:
        print("RESULT: fix the issues above (esp. TARGET_METER) before training.")
    print("=" * 70)
