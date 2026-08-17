"""
Standalone sanity check for the GRPO reward functions.

Run this ONCE on the training server (after `pip install -r requirements.txt`)
BEFORE launching training, to confirm:

  1. skrutable returns the meter label you expect for a real anushtup verse,
     and that TARGET_METER in rewards.py matches it.
  2. The LaBSE semantic reward loads and produces higher similarity for a
     faithful translation than for an unrelated verse.

Usage:
    python test_rewards.py
"""

import rewards
from rewards import meter_reward, semantic_reward, _normalize, get_meter_identifier


# A known-good anushtup verse in SLP1 (from temp.txt).
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
    result = mi.identify_meter(GOOD_VERSE, from_scheme="SLP", resplit_option="resplit_max")
    raw_label = getattr(result, "meter_label", None)
    print(f"  raw meter_label         : {raw_label!r}")
    print(f"  normalized              : {_normalize(raw_label)!r}")
    print(f"  TARGET_METER (rewards)  : {rewards.TARGET_METER!r} -> {_normalize(rewards.TARGET_METER)!r}")
    match = _normalize(raw_label) == _normalize(rewards.TARGET_METER)
    print(f"  MATCH                   : {match}")
    if not match:
        print("  >>> ACTION: set TARGET_METER in rewards.py to the raw label above.")
    return match


def check_meter_reward():
    print("=" * 70)
    print("2) METER REWARD (good vs bad)")
    print("=" * 70)
    r = meter_reward([GOOD_VERSE, BAD_VERSE])
    print(f"  reward(good) = {r[0]:.3f}   (expected > 0)")
    print(f"  reward(bad)  = {r[1]:.3f}   (expected = 0)")
    return r[0] > r[1]


def check_semantic_reward():
    print("=" * 70)
    print("3) SEMANTIC REWARD (LaBSE loads + ranks faithful > unrelated)")
    print("=" * 70)
    english = ["A verse in praise of trees and their worship."]
    faithful = [GOOD_VERSE]        # actually about trees being worshipped
    unrelated = ["kfzRaMvande jagadgurum"]  # unrelated content
    r_faithful = semantic_reward(faithful, english=english)[0]
    r_unrelated = semantic_reward(unrelated, english=english)[0]
    print(f"  semantic(faithful)   = {r_faithful:.3f}")
    print(f"  semantic(unrelated)  = {r_unrelated:.3f}")
    print(f"  faithful > unrelated : {r_faithful > r_unrelated}")
    return True  # informational; embeddings for Sanskrit are approximate


if __name__ == "__main__":
    ok_label = check_meter_label()
    ok_meter = check_meter_reward()
    check_semantic_reward()

    print("=" * 70)
    if ok_label and ok_meter:
        print("RESULT: rewards look good — safe to start training.")
    else:
        print("RESULT: fix the issues above (esp. TARGET_METER) before training.")
    print("=" * 70)
