"""Evaluate a model's meter reward on a separate list of English prompts."""

import argparse
import getpass
import json
import os
import random
import re
import statistics
from pathlib import Path

from inference_utils import generate_completions, load_model
from rewards import meter_reward


def default_model_path():
    output_root = os.environ.get(
        "GRPO_OUTPUT_ROOT", f"/tmp/{getpass.getuser()}/grpo/outputs"
    )
    return os.path.join(output_root, "chandomitra_gemma4_e4b_grpo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate anuṣṭubh meter rewards on unseen English prompts."
    )
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--label", default="trained")
    parser.add_argument("--dataset", default="sanganaka/anushtup")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--prompts", help="Optional text file to use instead of the dataset split."
    )
    parser.add_argument("--output", help="Destination JSONL path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of prompts to sample deterministically; use 0 for the full split.",
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_evaluation_rows(args):
    if args.prompts:
        lines = Path(args.prompts).read_text(encoding="utf-8").splitlines()
        rows = [
            {"source_index": index, "english": line.strip(), "reference_sanskrit": None}
            for index, line in enumerate(lines)
            if line.strip() and not line.startswith("#")
        ]
        source = str(Path(args.prompts))
        fingerprint = None
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, split=args.split)
        rows = [
            {
                "source_index": index,
                "english": example["English"].strip(),
                "reference_sanskrit": example["Sanskrit"].strip(),
            }
            for index, example in enumerate(dataset)
        ]
        source = f"{args.dataset}:{args.split}"
        fingerprint = dataset._fingerprint

    random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows, source, fingerprint


def default_output_path(label):
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "model"
    output_root = os.environ.get("GRPO_OUTPUT_ROOT", ".")
    return os.path.join(output_root, "evals", f"{safe_label}.jsonl")


def seed_generation(seed):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    evaluation_rows, source, fingerprint = load_evaluation_rows(args)
    if not evaluation_rows:
        raise ValueError(f"No evaluation prompts found in {source}")

    output_path = Path(args.output or default_output_path(args.label))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}", flush=True)
    model, processor, text_tokenizer = load_model(args.model)
    all_rewards = []

    with output_path.open("w", encoding="utf-8") as output_file:
        for prompt_index, evaluation_row in enumerate(evaluation_rows):
            english = evaluation_row["english"]
            seed_generation(args.seed + prompt_index)
            completions = generate_completions(
                model,
                processor,
                text_tokenizer,
                english,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            rewards = meter_reward(completions)
            all_rewards.extend(rewards)

            for generation_index, (completion, reward) in enumerate(
                zip(completions, rewards), start=1
            ):
                row = {
                    "label": args.label,
                    "model": args.model,
                    "prompt_index": prompt_index,
                    "source_index": evaluation_row["source_index"],
                    "generation_index": generation_index,
                    "english": english,
                    "reference_sanskrit": evaluation_row["reference_sanskrit"],
                    "completion": completion,
                    "meter_reward": reward,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            running_mean = statistics.fmean(all_rewards)
            print(
                f"[{prompt_index + 1}/{len(evaluation_rows)}] "
                f"mean_reward={running_mean:.4f}",
                flush=True,
            )

    summary = {
        "label": args.label,
        "model": args.model,
        "source": source,
        "source_fingerprint": fingerprint,
        "prompts": len(evaluation_rows),
        "generations": len(all_rewards),
        "mean_meter_reward": statistics.fmean(all_rewards),
        "perfect_rate": sum(score == 1.0 for score in all_rewards) / len(all_rewards),
        "nonzero_rate": sum(score > 0.0 for score in all_rewards) / len(all_rewards),
        "zero_rate": sum(score == 0.0 for score in all_rewards) / len(all_rewards),
        "seed": args.seed,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\nEvaluation summary")
    print(json.dumps(summary, indent=2))
    print(f"Rows: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()