# chandomitra — GRPO for Sanskrit anushtup poetry (Gemma 4 E4B)

GRPO fine-tuning of **`unsloth/gemma-4-E4B-it`** to generate Sanskrit verse in the
**anushtup** meter, rewarded on **syntactic** correctness (the verse must actually
scan as anuṣṭubh, verified by skrutable).

> Use the safetensors repo above, not `unsloth/gemma-4-E4B-it-GGUF`. GGUF is a
> llama.cpp inference format and cannot be LoRA-trained.

## Does this need a GPU?

**Yes — a CUDA GPU is required.** GRPO here uses Transformers generation for
rollouts plus bf16 LoRA training; neither runs meaningfully on CPU.

- **Known server profile:** 1× **44 GB** GPU, using a four-sequence rollout
  batch and one-sequence training micro-batches.
- **Comfortable:** 1× A100 80 GB or H100.

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
# 3. Optional Hugging Face authentication (the Unsloth checkpoint is public):
export HF_TOKEN=hf_xxx        # or: hf auth login
# 4. Optional: enable W&B logging (never hardcode the key):
wandb login
export GRPO_REPORT_TO=wandb   # default is "none"
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

The checkpoint contains a single 16 GB `model.safetensors` file. Ensure the
Hugging Face cache has at least 20 GB free. If the home filesystem is too small,
first list mounted filesystems and the common locations this account can write:

```bash
df -hT
for path in "$HOME" /tmp /scratch /data /mnt /work; do
  [ -d "$path" ] && [ -w "$path" ] && df -h "$path" | tail -n 1
done
```

Choose a writable location with at least 20 GB available, then verify it before
downloading. On the Sanskrit server, `/tmp` is on the 600 GB root filesystem and
currently has ample space, while `/home` is full. Keep every large artifact
under `/tmp`:

```bash
export GRPO_WORK_ROOT="/tmp/$USER/grpo"
export HF_HOME="$GRPO_WORK_ROOT/huggingface"
export GRPO_OUTPUT_ROOT="$GRPO_WORK_ROOT/outputs"
export WANDB_DIR="$GRPO_WORK_ROOT/wandb"
mkdir -p "$HF_HOME" "$GRPO_OUTPUT_ROOT" "$WANDB_DIR"
test -w "$GRPO_WORK_ROOT" && df -h "$GRPO_WORK_ROOT"

# Download first so network, cache, and disk errors are reported directly.
hf download unsloth/gemma-4-E4B-it
```

Files under `/tmp` may be deleted after a reboot or by periodic cleanup. Copy
the final adapter and any checkpoints you need to persistent storage after the
run. Use the same environment values whenever training is started.

Validate the reward and start training inside `tmux` so it survives an SSH
disconnect. Hugging Face and W&B login are not required for the default run:

```bash
python test_rewards.py

tmux new -s grpo
cd ~/grpo
source .venv/bin/activate
export GRPO_WORK_ROOT="/tmp/$USER/grpo"
export HF_HOME="$GRPO_WORK_ROOT/huggingface"
export GRPO_OUTPUT_ROOT="$GRPO_WORK_ROOT/outputs"
export WANDB_DIR="$GRPO_WORK_ROOT/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u train_grpo_gemma.py 2>&1 | tee "$GRPO_OUTPUT_ROOT/grpo_training.log"
```

To enable W&B, run `wandb login` once and export `GRPO_REPORT_TO=wandb` before
launching training. Leave that variable unset to disable W&B.

Detach from `tmux` with `Ctrl-b`, then `d`. Reconnect later with the same SSH
command and resume the terminal using:

```bash
tmux attach -t grpo
```

The run writes checkpoints, the merged model, the LoRA adapter, and W&B local
files beneath `GRPO_OUTPUT_ROOT`. If that variable is unset, outputs retain their
original locations beneath the current directory.

Gemma 4 must use `fast_inference=False` and `use_vllm=False` in this project.
Unsloth currently rejects Gemma 4's multimodal architecture on its vLLM path;
TRL instead performs the GRPO rollouts with the model's Transformers
`generate()` implementation.

The checked-in single-GPU settings deliberately use
`per_device_train_batch_size=1`, `generation_batch_size=4`, four generations,
and 128-token completions. `top_p=1.0` is also intentional: Transformers 5.5
implements top-p filtering by sorting the entire Gemma 4 vocabulary on every
generation step, which can request another 11+ GB of VRAM. `top_k=64` retains
truncated sampling without that full-vocabulary sort.

If loading reports that no `pytorch_model.bin` can be found, do not rename the
weights: this repository intentionally uses `model.safetensors`. Check the
pre-download command above. A failed or partial download, often caused by a full
cache filesystem, produces that generic Transformers error.

If loading fails because Transformers does not recognize model type `gemma4`,
the environment is too old. The checkpoint requires Transformers 5.5 or newer.
Upgrade the related packages together, verify the resolved versions, then retry:

```bash
source .venv/bin/activate
pip install --no-cache-dir --upgrade \
  "unsloth>=2026.8.18" "transformers>=5.5.0"
python -c "import transformers, unsloth; print('transformers', transformers.__version__); print('unsloth', unsloth.__version__)"
python -u train_grpo_gemma.py 2>&1 | tee "${GRPO_OUTPUT_ROOT:-.}/grpo_training.log"
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
