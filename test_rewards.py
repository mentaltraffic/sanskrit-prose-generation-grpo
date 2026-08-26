"""
Standalone sanity check for the GRPO reward functions.

Run this ONCE on the training server (after `pip install -r requirements.txt`)
BEFORE launching training, to confirm the reward grades a perfect anuṣṭubh as
1.0, a defective one in (0,1), garbage as 0.0, and off-scheme output poorly.

Usage:
    python test_rewards.py
"""

import math
import sys

import rewards
from rewards import (
    FORMAT_REWARD_WEIGHT,
    METER_REWARD_WEIGHT,
    TARGET_SCHEME,
    format_reward,
    format_score,
    get_meter_identifier,
    meter_reward,
    normalize_completion,
    normalized_meter_reward,
)

# Meter labels are Unicode; Windows consoles default to cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# A known-PERFECT anuṣṭubh verse in IAST (Bhagavad-gītā 1.1).
PERFECT_VERSE = (
    "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ "
    "māmakāḥ pāṇḍavāścaiva kimakurvata sañjaya"
)
# A real but metrically DEFECTIVE anuṣṭubh (from temp.txt; pada 2 is hypermetric).
GOOD_VERSE = (
    "aṅkolaṃkarajaṃplakṣaṃnyagrodhaṃjambucaamlakam ."
    "nīpaṃvānarayassarvepūjayāmāsurāhave .."
)
# Obviously non-metrical junk.
BAD_VERSE = "namaste world this is not a valid sanskrit verse at all"
# The same perfect verse in SLP1 — correct Sanskrit, wrong scheme.
OFF_SCHEME_VERSE = (
    "DarmakzetrekurukzetresamavetAyuyutsavaH "
    "mAmakAHpARqavAScEvakimakurvatasaMjaya"
)


def check_meter_label():
    print("=" * 70)
    print(f"1) METER SCAN (skrutable, from_scheme={TARGET_SCHEME})")
    print("=" * 70)
    mi = get_meter_identifier()
    graded = {}
    for name, verse in [
        ("perfect", PERFECT_VERSE),
        ("defective", GOOD_VERSE),
        ("garbage", BAD_VERSE),
    ]:
        result = mi.identify_meter(
            verse, from_scheme=TARGET_SCHEME, resplit_option="resplit_max"
        )
        graded[name] = rewards._score_meter_verse(result)
        print(f"  [{name}]")
        print(f"    meter_label : {getattr(result, 'meter_label', None)!r}")
        print(f"    is_perfect  : {getattr(result, 'is_perfect', None)}")
        print(f"    graded score: {graded[name]:.3f}")

    ok = graded["perfect"] == 1.0 and graded["garbage"] == 0.0
    print(f"  PERFECT==1.0 and GARBAGE==0.0 : {ok}")
    return ok


def check_total_reward():
    print("=" * 70)
    print("2) TOTAL REWARD (format + meter, as GRPO sees it)")
    print("=" * 70)
    verses = [PERFECT_VERSE, GOOD_VERSE, BAD_VERSE]
    formats = format_reward(verses)
    meters = meter_reward(verses)
    for name, fmt, met in zip(["perfect", "defective", "garbage"], formats, meters):
        print(f"  {name:<10} format={fmt:.3f}  meter={met:.3f}  total={fmt + met:.3f}")
    print(f"  (max format={FORMAT_REWARD_WEIGHT}, max meter={METER_REWARD_WEIGHT})")

    return math.isclose(formats[0] + meters[0], 1.0) and meters[2] == 0.0


def check_off_scheme():
    print("=" * 70)
    print("3) OFF-SCHEME OUTPUT (correct Sanskrit, wrong transliteration)")
    print("=" * 70)
    raw_format = format_score(OFF_SCHEME_VERSE)
    raw_meter = meter_reward([OFF_SCHEME_VERSE])[0]
    converted = normalize_completion(OFF_SCHEME_VERSE)
    converted_meter = normalized_meter_reward([OFF_SCHEME_VERSE])[0]
    print(f"  format_score     : {raw_format:.3f}   (expected < 1.0)")
    print(f"  meter_reward     : {raw_meter:.3f}   (expected low)")
    print(f"  converted        : {converted}")
    print(f"  normalized meter : {converted_meter:.3f}   (expected = 1.0)")

    # Off-scheme text must not out-earn honest in-scheme text.
    return raw_format < 1.0 and converted_meter == 1.0


def check_transliteration_normalization():
    print("=" * 70)
    print("4) ISO-15919 ANUSVARA VARIANT")
    print("=" * 70)
    iso_verse = "kamalaṁ jāgrahanti śayanaṁ"
    print(f"  format_score : {format_score(iso_verse):.3f}")
    print(f"  normalized   : {normalize_completion(iso_verse)}")
    # Dot-above anusvara must be folded to dot-below, not counted as invalid.
    return format_score(iso_verse) == 1.0


if __name__ == "__main__":
    results = [
        check_meter_label(),
        check_total_reward(),
        check_off_scheme(),
        check_transliteration_normalization(),
    ]

    print("=" * 70)
    if all(results):
        print("RESULT: rewards look good — safe to start training.")
    else:
        print(f"RESULT: fix the issues above (checks passed: {results}).")
    print("=" * 70)
