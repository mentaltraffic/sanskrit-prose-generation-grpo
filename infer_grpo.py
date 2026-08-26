"""Generate Sanskrit verse and report its meter reward."""

import argparse
import getpass
import os

from inference_utils import generate_completions, load_model
from rewards import (
    TARGET_SCHEME,
    format_score,
    is_target_format,
    normalize_completion,
    normalized_meter_reward,
)


def default_model_path():
    output_root = os.environ.get(
        "GRPO_OUTPUT_ROOT", f"/tmp/{getpass.getuser()}/grpo/outputs"
    )
    return os.path.join(output_root, "chandomitra_gemma4_e4b_grpo")


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Generate {TARGET_SCHEME} Sanskrit verse with the trained Gemma 4 model."
    )
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument(
        "--prompt",
        help="English meaning to render as verse; omit for an interactive session.",
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def generate_and_print(model, processor, text_tokenizer, english, args):
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    completions = generate_completions(
        model,
        processor,
        text_tokenizer,
        english,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    rewards = normalized_meter_reward(completions)
    for index, (completion, reward) in enumerate(zip(completions, rewards), start=1):
        print(
            f"\n[{index}] normalized_meter_reward={reward:.3f} "
            f"format_score={format_score(completion):.2f}"
        )
        print(completion)
        if not is_target_format(completion):
            print(f"{TARGET_SCHEME}: {normalize_completion(completion)}")


def main():
    args = parse_args()
    print(f"Loading model: {args.model}", flush=True)
    model, processor, text_tokenizer = load_model(args.model)

    if args.prompt:
        generate_and_print(model, processor, text_tokenizer, args.prompt, args)
        return

    print("Enter an English meaning. Submit an empty line to exit.")
    while True:
        try:
            english = input("\nEnglish> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not english:
            break
        generate_and_print(model, processor, text_tokenizer, english, args)


if __name__ == "__main__":
    main()