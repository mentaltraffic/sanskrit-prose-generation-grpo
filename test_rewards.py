"""
Standalone sanity check for the GRPO reward functions.

Run this ONCE on the training server (after `pip install -r requirements.txt`)
BEFORE launching training, to confirm skrutable grades a perfect anuṣṭubh as 1.0,
a defective one in (0,1), and garbage as 0.0.

Usage:
    python test_rewards.py
"""

import rewards
from rewards import meter_reward, get_meter_identifier


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


if __name__ == "__main__":
    ok_label = check_meter_label()
    ok_meter = check_meter_reward()
    ok_sem = check_semantic_reward()

    print("=" * 70)
    if ok_label and ok_meter and ok_sem:
        print("RESULT: rewards look good — safe to start training.")
    else:
        print("RESULT: fix the issues above (esp. TARGET_METER) before training.")
    print("=" * 70)
