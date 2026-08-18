# chandomitra — GRPO for Sanskrit anushtup poetry (Gemma 4 E4B)

GRPO fine-tuning of **`unsloth/gemma-4-E4B-it`** to generate Sanskrit verse in the
**anushtup** meter, rewarded on **syntactic** correctness (the verse must actually
scan as anuṣṭubh, verified by skrutable).

> Use the safetensors repo above, not `unsloth/gemma-4-E4B-it-GGUF`. GGUF is a
> llama.cpp inference format and cannot be LoRA-trained.

## Does this need a GPU?

**Yes — a CUDA GPU is required.** GRPO here uses Transformers generation for
rollouts plus bf16 LoRA training; neither runs meaningfully on CPU.

- **Minimum:** 1× GPU with **~24 GB** VRAM (e.g. RTX 3090/4090, A5000).
  Lower `per_device_train_batch_size` and `num_generations` in
  `train_grpo_gemma.py` if you hit OOM.
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
# 2. Rest of the deps (avoid filling a quota-limited server with pip's cache):
pip install --no-cache-dir -r requirements.txt
# 3. Gemma is a gated model on Hugging Face — authenticate:
export HF_TOKEN=hf_xxx        # or: hf auth login
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

Single-node multi-GPU is possible via `accelerate launch`, but the simplest
reliable setup is **one process with one GPU** doing both rollouts and training.
Scale up only after a single-GPU run works.

## Run on the Sanskrit server

Connect through the CNeRG jump host from PowerShell or another local terminal:

```bash
ssh -J monal-pg@cnerg.iitkgp.ac.in:8201 -p 8201 monal-pg@10.5.30.41
```

On the server, enter the project directory and verify that a CUDA GPU is visible:

```bash
cd ~/grpo                    # change this if the repository is elsewhere
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Pick the CUDA wheel index that matches the server driver/toolkit.
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt
```

Authenticate with Hugging Face (Gemma is gated), validate the reward, and start
training inside `tmux` so it survives an SSH disconnect:

```bash
hf auth login
wandb login                   # optional; required while report_to="wandb"
python test_rewards.py

tmux new -s grpo
source .venv/bin/activate
python -u train_grpo_gemma.py 2>&1 | tee grpo_training.log
```

Detach from `tmux` with `Ctrl-b`, then `d`. Reconnect later with the same SSH
command and resume the terminal using:

```bash
tmux attach -t grpo
```

The run writes checkpoints to `grpo_checkpoints_gemma4_e4b/`, the merged model
to `chandomitra_gemma4_e4b_grpo/`, and the LoRA adapter to
`chandomitra_gemma4_e4b_grpo_lora/`.

Gemma 4 must use `fast_inference=False` and `use_vllm=False` in this project.
Unsloth currently rejects Gemma 4's multimodal architecture on its vLLM path;
TRL instead performs the GRPO rollouts with the model's Transformers
`generate()` implementation.

If loading fails because Transformers does not recognize model type `gemma4`,
the environment is too old. The checkpoint requires Transformers 5.5 or newer.
Upgrade the related packages together, verify the resolved versions, then retry:

```bash
source .venv/bin/activate
pip install --no-cache-dir --upgrade \
  "unsloth>=2026.8.18" "transformers>=5.5.0"
python -c "import transformers, unsloth; print('transformers', transformers.__version__); print('unsloth', unsloth.__version__)"
python -u train_grpo_gemma.py 2>&1 | tee grpo_training.log
```

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
