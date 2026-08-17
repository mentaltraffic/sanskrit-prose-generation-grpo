# chandomitra — GRPO for Sanskrit anushtup poetry (Gemma 3 4B)

GRPO fine-tuning of **`unsloth/gemma-3-4b-it`** to generate Sanskrit verse in the
**anushtup** meter that is both **syntactically** (correct meter) and **semantically**
(faithful to the English meaning) correct.

## Does this need a GPU?

**Yes — a CUDA GPU is required.** GRPO here uses vLLM (`fast_inference=True`) for
rollouts plus bf16 LoRA training; neither runs meaningfully on CPU.

- **Minimum:** 1× GPU with **~24 GB** VRAM (e.g. RTX 3090/4090, A5000).
  Lower `gpu_memory_utilization`, `per_device_train_batch_size`, and
  `num_generations` in `train_grpo_gemma.py` if you hit OOM.
- **Comfortable:** 1× A100 40/80 GB or H100.
- The `sentence-transformers/LaBSE` embedder (semantic reward) also loads on GPU
  (~1.8 GB); it shares the same device.

## Files

| File | Purpose |
|------|---------|
| `train_grpo_gemma.py` | Main GRPO training script (Gemma 3 4B + `sanganaka/anushtup`). |
| `rewards.py` | Verifiable reward: skrutable **meter** check + **LaBSE** semantic similarity. Embedder isolated behind `get_embedder()`. |
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
```

## Run

```bash
python train_grpo_gemma.py
```

Single-node multi-GPU is possible via `accelerate launch`, but note **vLLM +
DDP is finicky**; the simplest reliable setup is **one process with one GPU**
doing both rollouts and training. Scale up only after a single-GPU run works.

## Reward design

The reference SFT scripts (`ref/train_ddp*.py`) do **not** check semantics or
meter at all — they only imitate a reference verse. GRPO needs a scalar reward,
so `rewards.py` introduces two, summed by TRL and logged separately:

1. **`meter_reward`** — feeds the generated SLP1 verse to skrutable's
   `MeterIdentifier` (see `ref/temp.txt`); `+1.0` when it scans as anushtup.
2. **`semantic_reward`** — transliterates the verse to Devanagari, embeds it and
   the English input with LaBSE, and rewards their cosine similarity.

Tune weights / target meter / embedder at the top of `rewards.py`.

> **Note on LaBSE:** Sanskrit isn't in LaBSE's official language list, so its
> Sanskrit embeddings are approximate. Swap the model inside `get_embedder()`
> (e.g. `intfloat/multilingual-e5-large`) without touching the reward logic.
