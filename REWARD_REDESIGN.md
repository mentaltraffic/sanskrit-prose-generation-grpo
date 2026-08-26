# Reward redesign — plan before any retrain

Context: the 2026-08-26 held-out evaluation showed GRPO raised the *hackable*
reward 10× (0.048 → 0.480) while the honest metre reward moved only 21%
(0.246 → 0.299), with **zero** perfect anuṣṭubh in 400 generations from either
the trained or the base model. See the run log in `README.md`.

Do these in order. Step 1 can invalidate everything after it.

## 1. Is the reward reachable at all? — ANSWERED: yes

`check_reward_ceiling.py` scores the dataset's own reference verses. Result on
200 verses of `sanganaka/anushtup:test`, identical for all three schemes:

| | SLP | IAST | DEV |
|---|---|---|---|
| mean score | 0.9980 | 0.9980 | 0.9980 |
| perfect (1.0) | 98.0% | 98.0% | 98.0% |
| zero (0.0) | 0.0% | 0.0% | 0.0% |

```bash
python check_reward_ceiling.py --scheme SLP --limit 200
```

**The reward is sound.** The scoring logic, the pāda rules and
`resplit_max` all work — ground-truth anuṣṭubh reaches `1.0`. So
`normalized_perfect_rate: 0.0` is a genuine model failure, not a measurement
artifact, and the retrain has a real target to aim at.

The gap is stark: humans 98% perfect, both models 0%.

## 2. Settle the transliteration scheme — DONE: IAST

Step 1 settles the scoring question: all three schemes score reference verses
*identically*, so scheme choice does **not** affect reward reachability.
skrutable normalises internally. The decision was made on model-side grounds:

- **IAST (chosen)** — what the model already emits, so the format problem
  disappears and the reward is dense from step one. skrutable's scansion output
  is always IAST regardless of input, so IAST is the library's native currency.
- **DEV** — the dataset's native form. Cost: the prompt states the metre rules as
  vowel *letters*, which appear as mātrās inside Devanagari syllables.
- **SLP1** — rejected. The scheme the model resists, the weakest for scheme
  detection on short strings, and the one that produced this failure.

`TARGET_SCHEME` in `rewards.py` is now the single switch.

## 3. Centralise the scheme and de-duplicate the prompt — DONE

`TARGET_SCHEME` drives the reward, the format check, the conversion helper and
`check_reward_ceiling.py`'s default. `HUMAN_PROMPT` now lives only in
`inference_utils.py`; `train_grpo_gemma.py` imports it, so training and
evaluation cannot drift apart.

## 4. Make the reward non-hackable *and* non-sparse — DONE

Strict format gating alone recreates the cold-start problem in reverse: if every
sample in a GRPO group scores `0.0`, the advantage is zero and there is no
gradient. The completed run already logged `frac_reward_zero_std: 0.25` with a
*permissive* reward.

Implemented as two separately-logged reward functions summing to 1.0:

- `format_reward` (0.3) — graded fraction of letters valid for `TARGET_SCHEME`,
  so a 90%-correct verse outranks a 40%-correct one and there is a slope to
  climb. Case-sensitive, since SLP1's capitals are what separate the schemes.
- `meter_reward` (0.7) — scanned as `TARGET_SCHEME`, so off-scheme output scores
  *lower* rather than higher. The failure mode is self-penalising, not
  exploitable.

Verified ordering in `test_rewards.py`: perfect IAST `1.000` > defective IAST
`0.825` > correct Sanskrit in the wrong scheme `0.556` > English `0.293`.

Known limitation: `format_reward` measures scheme validity, not Sanskrit-ness,
so lower-case English scores highly on format. It is a near-constant offset that
largely cancels in within-group advantage, and `meter_reward` discriminates.

Every assumption in a reward must be asserted or made impossible to violate.
`from_scheme='SLP'` was an unchecked precondition at the centre of a 65-hour
optimisation loop, and RL found the profitable violation.

## 5. Reconsider SFT → GRPO

`ref/train_ddp.py` put the SLP1 verse in the assistant turn, so cross-entropy
taught the scheme directly. The GRPO script keeps `reference_slp1` only for
logging, leaving one scalar per rollout as the sole channel for learning a
character encoding.

Standard recipe: SFT teaches the output format, GRPO refines metre. Given zero
perfect verses from either model, metre alone may be too hard for pure RL from
this starting point.

## 6. Paired significance test before quoting +21%

100 prompts × 4 correlated samples, reward std ≈ 0.3, so +0.053 is roughly
1.5–2 standard errors — suggestive, not conclusive. Both runs used identical
prompts and seeds, so a paired per-prompt test on the existing JSONL rows
(`source_index` is recorded) is both more powerful and more honest than
comparing means.
