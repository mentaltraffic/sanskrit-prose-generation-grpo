"""Step 1 of REWARD_REDESIGN.md: can the reward reach 1.0 on ground-truth verses?

Scores the dataset's own reference verses. If real anuṣṭubh does not score 1.0,
the objective is unreachable and no amount of training fixes it.
"""

import argparse
import collections
import statistics
import sys

import rewards

# Meter labels are Unicode; Windows consoles default to cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# skrutable scheme name -> indic_transliteration target for the Devanagari source.
SCHEMES = {"DEV": None, "SLP": "slp1", "IAST": "iast"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sanganaka/anushtup")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scheme", default=rewards.TARGET_SCHEME, choices=sorted(SCHEMES))
    parser.add_argument("--resplit-option", default="resplit_max")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--show-failures", type=int, default=3)
    return parser.parse_args()


def convert(devanagari, scheme):
    target = SCHEMES[scheme]
    if target is None:
        return devanagari
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    return transliterate(devanagari, sanscript.DEVANAGARI, target)


def main():
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    identifier = rewards.get_meter_identifier()

    scores = []
    labels = collections.Counter()
    failures = []

    for example in dataset:
        if args.limit and len(scores) >= args.limit:
            break
        source = (example["Sanskrit"] or "").strip()
        if not source:
            continue

        verse_text = convert(source, args.scheme)
        try:
            verse = identifier.identify_meter(
                verse_text,
                from_scheme=args.scheme,
                resplit_option=args.resplit_option,
            )
            score = rewards._score_meter_verse(verse)
            label = (getattr(verse, "meter_label", "") or "")[:60]
        except Exception as error:
            score = 0.0
            label = f"ERROR: {type(error).__name__}"

        scores.append(score)
        labels[label] += 1
        if score < 1.0 and len(failures) < args.show_failures:
            failures.append((score, label, verse_text[:110]))

    if not scores:
        raise SystemExit("No reference verses scored.")

    perfect = sum(score == 1.0 for score in scores)
    zero = sum(score == 0.0 for score in scores)

    print("=" * 70)
    print(f"REWARD CEILING — {args.dataset}:{args.split}")
    print(f"scheme={args.scheme}  resplit={args.resplit_option}  n={len(scores)}")
    print("=" * 70)
    print(f"  mean score   : {statistics.fmean(scores):.4f}")
    print(f"  perfect (1.0): {perfect / len(scores):.1%}  ({perfect})")
    print(f"  zero    (0.0): {zero / len(scores):.1%}  ({zero})")

    print("\n  score distribution:")
    for score, count in sorted(collections.Counter(scores).items()):
        print(f"    {score:.2f}  {'#' * int(40 * count / len(scores)):<40} {count}")

    print("\n  most common meter labels:")
    for label, count in labels.most_common(5):
        print(f"    {count:>4}  {label}")

    if failures:
        print("\n  example non-perfect reference verses:")
        for score, label, text in failures:
            print(f"    score={score:.2f}  label={label}")
            print(f"      {text}")

    print("\n" + "=" * 70)
    if perfect / len(scores) < 0.5:
        print("VERDICT: ground-truth verses mostly do NOT score 1.0.")
        print("The reward ceiling is broken — fix scoring before retraining.")
    else:
        print("VERDICT: the reward is reachable on real verses.")
    print("=" * 70)


if __name__ == "__main__":
    main()
