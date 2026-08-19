"""
Diagnostic: isolate the `prob_dist must be 1 or 2 dim` GRPO failure.

Bypasses TRL entirely and drives the model directly, so the extra tensor axis is
attributed to either the model/generation stack or to TRL's input construction.

Run on the Sanskrit server with the same env vars used for training:
    python -u debug_generate.py
"""

import os

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_CACHE"] = "1"

import torch
from unsloth import FastModel

MODEL_NAME = "unsloth/gemma-4-E4B-it"

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=False,
    use_safetensors=True,
    fast_inference=False,
    device_map="auto",
    use_gradient_checkpointing="unsloth",
)

text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
text_tokenizer.padding_side = "left"
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

print("=" * 70)
print("tokenizer type      :", type(tokenizer).__name__)
print("text_tokenizer type :", type(text_tokenizer).__name__)
print("model type          :", type(model).__name__)
print("base model type     :", type(getattr(model, "base_model", model)).__name__)

messages = [{"role": "user", "content": "Say one short sentence."}]
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 1) Input rank straight from the tokenizer.
encoded = text_tokenizer([prompt_text], return_tensors="pt", padding=True)
for key, value in encoded.items():
    if isinstance(value, torch.Tensor):
        print(f"encoded[{key!r}].shape :", tuple(value.shape))

device = next(model.parameters()).device
encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}

# 2) Logits rank from a plain forward pass (no generation machinery).
print("-" * 70)
with torch.inference_mode():
    out = model(**encoded)
print("forward logits.shape :", tuple(out.logits.shape))
print("   -> expected rank 3 (batch, seq, vocab)")

# 3) Same forward with the logits_to_keep=1 that generation injects.
print("-" * 70)
try:
    with torch.inference_mode():
        out_keep = model(**encoded, logits_to_keep=1)
    print("logits_to_keep=1 logits.shape :", tuple(out_keep.logits.shape))
except Exception as exc:
    print("logits_to_keep=1 raised :", type(exc).__name__, exc)

# 4) Sampling generation with loose kwargs (the known-good baseline).
print("-" * 70)
try:
    with torch.inference_mode():
        gen = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=True,
            temperature=1.0,
            top_k=64,
            top_p=1.0,
        )
    print("loose-kwargs generate OK, sequences.shape :", tuple(gen.shape))
except Exception as exc:
    print("loose-kwargs generate raised :", type(exc).__name__, exc)

# Report tensor ranks at the model boundary during generation.
_hooks = []


def _pre_hook(_module, _args, kwargs):
    ids = kwargs.get("input_ids")
    embeds = kwargs.get("inputs_embeds")
    ltk = kwargs.get("logits_to_keep")
    print(
        "  forward in : input_ids=",
        tuple(ids.shape) if isinstance(ids, torch.Tensor) else None,
        "| inputs_embeds=",
        tuple(embeds.shape) if isinstance(embeds, torch.Tensor) else None,
        "| logits_to_keep=",
        f"{type(ltk).__name__}"
        + (f"{tuple(ltk.shape)}" if isinstance(ltk, torch.Tensor) else f"({ltk})"),
    )


def _post_hook(_module, _args, _kwargs, output):
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        print("  forward out: logits=", tuple(logits.shape), "rank", logits.ndim)


_hooks.append(model.register_forward_pre_hook(_pre_hook, with_kwargs=True))
_hooks.append(model.register_forward_hook(_post_hook, with_kwargs=True))

# 5) Recover the exact GenerationConfig that TRL builds from GRPOConfig.
print("-" * 70)
trainer_generation_config = None
try:
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    from rewards import REWARD_FUNCS

    probe_rows = load_dataset("sanganaka/anushtup", split="train[:8]")
    probe_rows = probe_rows.map(
        lambda ex: {"prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["English"]}],
            tokenize=False,
            add_generation_prompt=True,
        )},
        remove_columns=probe_rows.column_names,
    )

    probe_args = GRPOConfig(
        use_vllm=False,
        num_generations=4,
        generation_batch_size=4,
        max_prompt_length=1024,
        max_completion_length=128,
        temperature=1.0,
        top_p=1.0,
        top_k=64,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        report_to="none",
        output_dir="/tmp/grpo_probe",
    )
    probe_trainer = GRPOTrainer(
        model=model,
        processing_class=text_tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=probe_args,
        train_dataset=probe_rows,
    )
    trainer_generation_config = probe_trainer.generation_config
    print("TRL generation_config:")
    for key, value in sorted(trainer_generation_config.to_diff_dict().items()):
        print(f"    {key} = {value!r}")
except Exception as exc:
    print("could not build trainer :", type(exc).__name__, exc)

# 6) The decisive call: TRL's own config, exactly as _generate_single_turn issues it.
print("-" * 70)
if trainer_generation_config is not None:
    try:
        with torch.inference_mode():
            gen2 = model.generate(
                **encoded,
                generation_config=trainer_generation_config,
                disable_compile=True,
            )
        print("TRL-config generate OK, sequences.shape :", tuple(gen2.shape))
    except Exception as exc:
        print("TRL-config generate raised :", type(exc).__name__, exc)

for hook in _hooks:
    hook.remove()

print("=" * 70)
print(
    "Step 4 passing while step 6 raises isolates the fault to TRL's GenerationConfig;\n"
    "the forward-hook lines show which tensor gains the extra axis."
)
