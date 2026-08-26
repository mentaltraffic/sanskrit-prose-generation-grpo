"""Shared Gemma 4 loading and generation helpers for inference and evaluation."""

import functools
import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_CACHE", "1")

HUMAN_PROMPT = """
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


def _patch_generation(model):
    """Prevent Gemma 4 from adding an extra logits dimension during generation."""
    generation_model = getattr(model, "base_model", model)
    generation_model = getattr(generation_model, "model", generation_model)
    original_prepare_inputs = generation_model.prepare_inputs_for_generation

    @functools.wraps(original_prepare_inputs)
    def prepare_inputs_with_logits_to_keep(*args, **kwargs):
        model_inputs = original_prepare_inputs(*args, **kwargs)
        if model_inputs.get("logits_to_keep") is None:
            model_inputs["logits_to_keep"] = 1
        return model_inputs

    generation_model.prepare_inputs_for_generation = prepare_inputs_with_logits_to_keep


def load_model(model_name):
    """Load a merged model, LoRA directory, or Hugging Face base model."""
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=False,
        use_safetensors=True,
        fast_inference=False,
        device_map="auto",
    )
    text_tokenizer = getattr(processor, "tokenizer", processor)
    text_tokenizer.padding_side = "left"
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    _patch_generation(model)
    model.eval()
    return model, processor, text_tokenizer


def build_prompt(processor, english):
    """Apply the same rules and chat template used during GRPO training."""
    messages = [{"role": "user", "content": HUMAN_PROMPT.format(english.strip())}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_completions(
    model,
    processor,
    text_tokenizer,
    english,
    num_generations=1,
    max_new_tokens=128,
    temperature=1.0,
    top_k=64,
):
    """Generate and clean one or more SLP1 verse candidates."""
    import torch

    from rewards import _clean_completion

    prompt = build_prompt(processor, english)
    encoded = text_tokenizer([prompt], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=1.0,
            num_return_sequences=num_generations,
            pad_token_id=text_tokenizer.pad_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
        )

    completion_ids = output_ids[:, prompt_length:]
    decoded = text_tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [_clean_completion(text) for text in decoded]