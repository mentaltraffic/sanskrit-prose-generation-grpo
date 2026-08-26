"""
GRPO fine-tuning of Gemma 4 E4B for Sanskrit (anushtup) poetry generation.

- Base model : unsloth/gemma-4-E4B-it
- Dataset    : sanganaka/anushtup  (English meaning -> Sanskrit anushtup verse)
- Reward     : verifiable skrutable meter check (syntactic correctness) from rewards.py

Scaffolding (env vars / device map / wandb / DDP flags) mirrors train_ddp_dev.py.
Unlike the SFT scripts, the model is optimised with GRPO instead of
cross-entropy on a reference verse. Gemma 4 uses Transformers generation because
Unsloth does not currently support fast_inference (vLLM) for this architecture.

NOTE: This requires a CUDA GPU (Transformers generation + bf16 training).
See README.md.
Launch:  python train_grpo_gemma.py
     or  accelerate launch train_grpo_gemma.py   (single-node; see README caveats)
"""

import os

OUTPUT_ROOT = os.path.abspath(os.environ.get("GRPO_OUTPUT_ROOT", "."))
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- wandb -----------------------------------------------------------------
# Do NOT hardcode your API key. Authenticate on the machine instead, e.g.:
#   export WANDB_API_KEY=xxxx      (or run `wandb login`)
# Set GRPO_REPORT_TO=wandb to opt in; logging is disabled by default.
REPORT_TO = os.environ.get("GRPO_REPORT_TO", "none")
os.environ.setdefault("WANDB_PROJECT", "chandomitra")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
os.environ.setdefault("WANDB_DIR", os.path.join(OUTPUT_ROOT, "wandb"))

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
    print(f"training outputs: {OUTPUT_ROOT}")
    print(f"reporting integration: {REPORT_TO}")

import torch

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_CACHE"] = "1"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/gemma-4-E4B-it"
max_seq_length = 2048
max_prompt_length = 1024          # the rules prompt is long
max_completion_length = 128       # enough for a 32-syllable anushtup verse
lora_rank = 32

# Gemma 4 is multimodal, so it loads through FastModel rather than FastLanguageModel.
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    use_safetensors=True,
    fast_inference=False,         # Gemma 4 is not supported by Unsloth's vLLM path
    device_map=get_device_map(),
    use_gradient_checkpointing="unsloth",
    # token = "hf_...",           # optional for this public checkpoint
)

# For multimodal checkpoints FastModel hands back a processor; the text tokenizer is nested.
text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

if IS_MAIN_PROCESS:
    print(text_tokenizer.padding_side)

model = FastModel.get_peft_model(
    model,
    r=lora_rank,
    finetune_vision_layers=False,     # this task is text-only
    finetune_language_layers=True,
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
text_tokenizer.padding_side = "left"
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# Gemma 4 indexes hidden_states with logits_to_keep, where None inserts an axis and
# yields rank-4 logits. Unsloth's patched forward hides the kwarg, so generation
# leaves it None; pin it to the last position instead.
import functools

_gen_model = getattr(model, "base_model", model)
_gen_model = getattr(_gen_model, "model", _gen_model)
_orig_prepare_inputs = _gen_model.prepare_inputs_for_generation


# wraps() keeps the signature that generate() inspects to validate model kwargs.
@functools.wraps(_orig_prepare_inputs)
def _prepare_inputs_with_logits_to_keep(*args, **kwargs):
    model_inputs = _orig_prepare_inputs(*args, **kwargs)
    if model_inputs.get("logits_to_keep") is None:
        model_inputs["logits_to_keep"] = 1
    return model_inputs


_gen_model.prepare_inputs_for_generation = _prepare_inputs_with_logits_to_keep

# ---------------------------------------------------------------------------
# Dataset  (same source + rules prompt as train_ddp.py)
# ---------------------------------------------------------------------------
from datasets import load_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from inference_utils import build_prompt as build_prompt_text

dataset = load_dataset("sanganaka/anushtup")


def build_prompt(example):
    english = example["English"].strip()
    # Reference verse kept only for logging / optional eval, not for the reward.
    reference = transliterate(
        example["Sanskrit"], sanscript.DEVANAGARI, sanscript.IAST
    ).strip()

    return {
        # Built by inference_utils so training and evaluation cannot drift apart.
        "prompt": build_prompt_text(tokenizer, english),
        "english": english,      # kept for logging/analysis (not used by the reward)
        "reference": reference,
    }


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

per_device_train_batch_size = 1
num_generations = 4                      # completions sampled per prompt
generation_batch_size = num_generations * WORLD_SIZE
gradient_accumulation_steps = int(16 / per_device_train_batch_size / WORLD_SIZE) or 1

# One generation round per prompt without vLLM, so a full epoch over the 8,306-row
# train split is ~8.3k rollouts. Cap the budget instead of walking off a cliff.
num_train_epochs = float(os.environ.get("GRPO_EPOCHS", "1"))
max_steps = int(os.environ.get("GRPO_MAX_STEPS", "-1"))

training_args = GRPOConfig(
    # generation
    use_vllm=False,
    num_generations=num_generations,
    generation_batch_size=generation_batch_size,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    # Avoid top-p's full-vocabulary sort, which peaks above 44 GB on Gemma 4.
    temperature=1.0,
    top_p=1.0,
    top_k=64,
    # optimisation
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
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
    report_to=REPORT_TO,
    run_name="gemma4_e4b_grpo_anushtup",
    output_dir=os.path.join(OUTPUT_ROOT, "grpo_checkpoints_gemma4_e4b"),
    seed=3407,
    # DDP
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = GRPOTrainer(
    model=model,
    # The Gemma 4 processor emits multimodal ids/extra axes that break text-only sampling.
    processing_class=text_tokenizer,
    reward_funcs=REWARD_FUNCS,
    args=training_args,
    train_dataset=train_dataset,
    # GRPOTrainer uses eval_dataset only if eval_strategy is set; kept for convenience.
    eval_dataset=eval_dataset,
)

if IS_MAIN_PROCESS:
    print("Starting GRPO trainer.train")

# Set GRPO_DEBUG_SHAPES=1 to print tensor ranks at the model boundary.
if os.environ.get("GRPO_DEBUG_SHAPES") == "1":
    _inner = getattr(model, "base_model", model)
    _inner = getattr(_inner, "model", _inner)
    _calls = {"n": 0}

    def _dbg_pre(_module, _args, kwargs):
        if _calls["n"] >= 8:
            return
        ids = kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        ltk = kwargs.get("logits_to_keep")
        print(
            "[shapes] in : input_ids=",
            tuple(ids.shape) if isinstance(ids, torch.Tensor) else None,
            "| inputs_embeds=",
            tuple(embeds.shape) if isinstance(embeds, torch.Tensor) else None,
            "| logits_to_keep=",
            f"{type(ltk).__name__}"
            + (f"{tuple(ltk.shape)}" if isinstance(ltk, torch.Tensor) else f"({ltk})"),
            flush=True,
        )

    def _dbg_post(_module, _args, _kwargs, output):
        if _calls["n"] >= 8:
            return
        _calls["n"] += 1
        logits = getattr(output, "logits", None)
        print(
            "[shapes] out: logits=",
            tuple(logits.shape) if isinstance(logits, torch.Tensor) else type(logits).__name__,
            flush=True,
        )

    _inner.register_forward_pre_hook(_dbg_pre, with_kwargs=True)
    _inner.register_forward_hook(_dbg_post, with_kwargs=True)
    print(f"[shapes] hooks attached to {type(_inner).__name__}", flush=True)

trainer.train()

if IS_MAIN_PROCESS:
    model.save_pretrained_merged(
        os.path.join(OUTPUT_ROOT, "chandomitra_gemma4_e4b_grpo"),
        tokenizer,
        save_method="merged_16bit",
    )
    # LoRA-only adapter (small, handy for resuming / sharing):
    model.save_lora(os.path.join(OUTPUT_ROOT, "chandomitra_gemma4_e4b_grpo_lora"))


if __name__ == "__main__":
    print("GRPO training completed successfully!")
