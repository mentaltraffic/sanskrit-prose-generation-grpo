"""
GRPO fine-tuning of Gemma 3 4B for Sanskrit (anushtup) poetry generation.

- Base model : unsloth/gemma-3-4b-it
- Dataset    : sanganaka/anushtup  (English meaning -> Sanskrit anushtup verse)
- Reward     : verifiable, combined signal from rewards.py
                 * skrutable meter check   (syntactic correctness)
                 * LaBSE embedding sim.    (semantic correctness)

Scaffolding (env vars / device map / wandb / DDP flags) mirrors train_ddp_dev.py.
Unlike the SFT scripts, generation uses Unsloth's fast_inference (vLLM) and the
model is optimised with GRPO instead of cross-entropy on a reference verse.

NOTE: This requires a CUDA GPU (vLLM + bf16 training). See README.md.
Launch:  python train_grpo_gemma.py
     or  accelerate launch train_grpo_gemma.py   (single-node; see README caveats)
"""

import os

# --- wandb -----------------------------------------------------------------
os.environ["WANDB_API_KEY"] = "c8eab488840dc53cdb56f1d5d38e37d2582f42d3"
os.environ["WANDB_PROJECT"] = "chandomitra"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# --- process / device bookkeeping (same pattern as train_ddp_dev.py) -------
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
IS_MAIN_PROCESS = LOCAL_RANK in [-1, 0]


def get_device_map() -> "str | dict[str, int]":
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return "auto"
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    torch.cuda.set_device(local_rank)
    return {"": local_rank}


if IS_MAIN_PROCESS:
    print(f"gpus available: {WORLD_SIZE}")

import torch

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_CACHE"] = "1"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/gemma-3-4b-it"
max_seq_length = 2048
max_prompt_length = 1024          # the rules prompt is long
max_completion_length = 256       # an anushtup verse is short
lora_rank = 32

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,          # enable vLLM generation for GRPO rollouts
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,   # lower if you hit OOM during vLLM init
    device_map=get_device_map(),
    use_gradient_checkpointing="unsloth",
    # token = "hf_...",           # gemma is gated on HF; export HF_TOKEN or pass here
)

if IS_MAIN_PROCESS:
    print(tokenizer.padding_side)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    loftq_config=None,
)

# GRPO decodes; left padding is correct for generation.
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# ---------------------------------------------------------------------------
# Dataset  (same source + rules prompt as train_ddp.py)
# ---------------------------------------------------------------------------
from datasets import load_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

dataset = load_dataset("sanganaka/anushtup")

human_prompt = """
### INSTRUCTION:
The goal is to generate Sanskrit verse that follows the anushtup meter rules for the given input translation.
RULES:
Verse Rules:
The verse contains 32 syllables/akshara and 4 padas in total.
The verse is divided in 2 lines, each containing 16 syllables.
Each line is divided into 2 padas (quartets),each containing exactly 8 syllables.
The fifth syllable of every pada must be LAGHU or short.
The sixth syllable of every pada must be GURU or long.
The seventh syllable of the second and the fourth pada must be HRASVA
The seventh letter of the first and third paada must be DEERGHA

Syllable Rules:
LAGHU vowels: a, i, u, f, x
GURU vowels: A, I, U, F, X, e, E, o, O
HRASVA vowels: a, i, u, f, x
DEERGHA vowels: A, I, U, F, X, e, E, o, O

Syllable is marked laghu, guru and hrasva, deergha based on the vowel it contains.
Syllable containing anusvAra("M") or visarga("H") is always marked as guru.
Syllable that is followed by a conjunct consonant (saMyuktAkzara) is always marked guru.

Respond ONLY with the Sanskrit verse in SLP1 transliteration, nothing else.

### INPUT:
{}
### RESPONSE:
"""


def build_prompt(example):
    english = example["English"].strip()
    # Reference verse (SLP1) kept only for logging / optional eval, not for the reward.
    reference_slp1 = transliterate(
        example["Sanskrit"], sanscript.DEVANAGARI, sanscript.SLP1
    ).strip()

    messages = [{"role": "user", "content": human_prompt.format(english)}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt_text,   # TRL feeds this to the model
        "english": english,      # forwarded to reward fns (semantic reward)
        "reference": reference_slp1,
    }


keep_cols = ["prompt", "english", "reference"]
train_dataset = dataset["train"].map(
    build_prompt, remove_columns=dataset["train"].column_names
)
eval_split = "test" if "test" in dataset else ("valid" if "valid" in dataset else None)
eval_dataset = (
    dataset[eval_split].map(build_prompt, remove_columns=dataset[eval_split].column_names)
    if eval_split
    else None
)

# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from rewards import REWARD_FUNCS

per_device_train_batch_size = 4          # must be divisible by num_generations
num_generations = 4                      # completions sampled per prompt
gradient_accumulation_steps = int(16 / per_device_train_batch_size / WORLD_SIZE) or 1

training_args = GRPOConfig(
    # generation
    use_vllm=True,
    num_generations=num_generations,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    temperature=1.0,
    # optimisation
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=3,
    learning_rate=5e-6,
    warmup_ratio=0.08,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    beta=0.04,                            # KL penalty vs. the reference policy
    # precision
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    # logging / checkpointing
    logging_steps=1,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    report_to="wandb",
    run_name="gemma3_4b_grpo_anushtup",
    output_dir="grpo_checkpoints_gemma3_4b",
    seed=3407,
    # DDP
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=REWARD_FUNCS,
    args=training_args,
    train_dataset=train_dataset,
    # GRPOTrainer uses eval_dataset only if eval_strategy is set; kept for convenience.
    eval_dataset=eval_dataset,
)

if IS_MAIN_PROCESS:
    print("Starting GRPO trainer.train")

trainer.train()

if IS_MAIN_PROCESS:
    model.save_pretrained_merged(
        "chandomitra_gemma3_4b_grpo", tokenizer, save_method="merged_16bit"
    )
    # LoRA-only adapter (small, handy for resuming / sharing):
    model.save_lora("chandomitra_gemma3_4b_grpo_lora")


if __name__ == "__main__":
    print("GRPO training completed successfully!")
