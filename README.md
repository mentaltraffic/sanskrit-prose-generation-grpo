# chandomitra — GRPO for Sanskrit anushtup poetry (Gemma 4 E4B)

GRPO fine-tuning of **`unsloth/gemma-4-E4B-it`** to generate Sanskrit verse in the
**anushtup** meter, rewarded on **syntactic** correctness (the verse must actually
scan as anuṣṭubh, verified by skrutable).

> Use the safetensors repo above, not `unsloth/gemma-4-E4B-it-GGUF`. GGUF is a
> llama.cpp inference format and cannot be LoRA-trained.

## Does this need a GPU?

**Yes — a CUDA GPU is required.** GRPO here uses vLLM (`fast_inference=True`) for
rollouts plus bf16 LoRA training; neither runs meaningfully on CPU.

- **Minimum:** 1× GPU with **~24 GB** VRAM (e.g. RTX 3090/4090, A5000).
  Lower `gpu_memory_utilization`, `per_device_train_batch_size`, and
  `num_generations` in `train_grpo_gemma.py` if you hit OOM.
- **Comfortable:** 1× A100 40/80 GB or H100.

## Files

| File | Purpose |
|------|---------|
| `train_grpo_gemma.py` | Main GRPO training script (Gemma 4 E4B + `sanganaka/anushtup`). |
| `rewards.py` | Verifiable **meter** reward via skrutable (deterministic anuṣṭubh scan check). |
| `test_rewards.py` | Sanity check for the reward fn; run before training. |
| `requirements.txt` | Python dependencies. |
| `ref/` | Original SFT scripts kept for reference (`train_ddp.py`, `train_ddp_dev.py`, `temp.txt`). |

## Setup

```bash
# 1. Install a CUDA torch build matching your driver, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2. Rest of the deps:
pip install -r requirements.txt
# 3. Gemma is a gated model on Hugging Face — authenticate:
export HF_TOKEN=hf_xxx        # or: huggingface-cli login
# 4. Optional: enable W&B logging (never hardcode the key):
export WANDB_API_KEY=xxxx     # or: wandb login   (else set report_to="none")
```

## Validate rewards before training

```bash
python test_rewards.py
```

Confirms skrutable grades a perfect anuṣṭubh (Gītā 1.1) as `1.0`, a defective
verse in `(0,1)`, and garbage as `0.0`. If the printed `meter_label` no longer
starts with anuṣṭubh, update `TARGET_METER` in `rewards.py`.

## Run

```bash
python train_grpo_gemma.py
```

Single-node multi-GPU is possible via `accelerate launch`, but note **vLLM +
DDP is finicky**; the simplest reliable setup is **one process with one GPU**
doing both rollouts and training. Scale up only after a single-GPU run works.

## Reward design

The reference SFT scripts (`ref/train_ddp*.py`) do **not** check meter at all —
they only imitate a reference verse. GRPO needs a scalar reward, so `rewards.py`
provides one **syntactic** signal (semantics were intentionally dropped):

**`meter_reward`** — feeds the generated SLP1 verse to skrutable's
`MeterIdentifier` (see `ref/temp.txt`) and grades it deterministically from the
guru/laghu `syllable_weights` grid rather than the permissive prose label:

- `1.0` — skrutable flags a **perfect** anuṣṭubh (`is_perfect`).
- `(0, 0.9]` — anuṣṭubh family; partial credit per pāda for correct 8-syllable
  length and the 5th-laghu / 6th-guru / 7th (guru in odd pādas, laghu in even) rule.
- `0.0` — not anuṣṭubh (or unparsable garbage).

Tune weights / target meter at the top of `rewards.py`.
