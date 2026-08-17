import os

# determine if on main process + Turn off unsloth RL patches:
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
IS_MAIN_PROCESS = LOCAL_RANK in [-1,0]

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
os.environ['UNSLOTH_COMPILE_DISABLE'] = "1"
os.environ['UNSLOTH_DISABLE_CACHE'] = "1"
os.environ['UNSLOTH_DISABLERL_PATCH'] = "1"

from datasets import load_dataset
dataset = load_dataset("csv", data_files={"train":"dataset/hindi-ip/train.csv", "valid": "dataset/hindi-ip/val.csv", "test": "dataset/hindi-ip/test.csv"})

MODEL_NAME = "unsloth/phi-4"

from unsloth import FastLanguageModel  # FastVisionModel for LLMs
import torch
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    device_map = get_device_map(), # device_map={"": torch.cuda.current_device()},
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

 
if IS_MAIN_PROCESS:
    print(tokenizer.padding_side)
    print(model)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    # finetune_vision_layers     = False, # Turn off for just text!
    # finetune_language_layers   = True,  # Should leave on!
    # finetune_attention_modules = True,  # Attention good for GRPO
    # finetune_mlp_modules       = True,  # SHould leave on always!
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    loftq_config = None, # And LoftQ
)

# if IS_MAIN_PROCESS:
#     os.environ['WANDB_DISABLED'] = "false"

# else:
#     import wandb

#     wandb.login()

#     wandb.init(
#         project="chandomitra",
#         entity="manojbalaji1",
#         name="ddp_phi4"
#     )

tokenizer.padding_side = "left"   # causal LM
tokenizer.pad_token = tokenizer.eos_token

per_device_train_batch_size = 1
gradient_accumalation_steps = int(64 / per_device_train_batch_size / WORLD_SIZE)
epochs = 20
learning_rate = 1e-5

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
)


phi4_prompt = """
<|im_start|>user<|im_sep|>
{}
Input: {}
Output:<|im_end|><|im_start|>assistant<|im_sep|>
{}
"""

human_prompt = """{} Input:{} Output:"""


EOS_TOKEN = tokenizer.eos_token
# Define your function
def preprocess(example):
    conversations = [{"role": "user", "content": human_prompt.format(example['prompt'].strip(), example["meaning"].strip())},
    {"role": "assistant", "content": example['Shlok'].strip()}]
    # texts = tokenizer.apply_chat_template(
    #         convo, tokenize = False, add_generation_prompt = False
    #     )
    example["conversations"] = conversations
    return example


# Apply the function
dataset = dataset.map(preprocess)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return { "text" : texts, }


new_dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

import os
os.environ['WANDB_API_KEY'] = 'c8eab488840dc53cdb56f1d5d38e37d2582f42d3'
os.environ["WANDB_PROJECT"] = "chandomitra"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# --- keep all your dataset/tokenizer/model code the same up to trainer creation ---

from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq


from unsloth import is_bfloat16_supported 
training_args = SFTConfig(
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumalation_steps,
        num_train_epochs = 10,
        warmup_steps = 40,

        logging_strategy="steps",
        save_strategy = "steps",
        save_steps = 10,
        save_total_limit = 3,
        eval_strategy = "steps",
        eval_steps = 10,
        logging_steps = 1,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        report_to = "wandb",
        run_name = "phi4_hindi-ip_response_only_ddp_2gpu",

        bf16 = is_bfloat16_supported(),
        fp16= not is_bfloat16_supported(),
        output_dir = "training_checkpoints_response",
        
        learning_rate = 1e-6,
        optim = "adamw_torch",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 3407,

        # DDP args
        ddp_find_unused_parameters=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    )


from transformers import EarlyStoppingCallback
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = new_dataset['train'],
    eval_dataset = new_dataset['valid'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False,
    args = training_args,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001,
        )
    ]
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

# ============================================================================
# EARLY STOPPING
# ============================================================================

# from transformers import EarlyStoppingCallback

# early_stopping_callback = EarlyStoppingCallback(
#     early_stopping_patience=2,
#     early_stopping_threshold=0.001,
# )

# trainer.add_callback(early_stopping_callback)


if IS_MAIN_PROCESS:
    print("Starting trainer.train")
trainer_stats = trainer.train(resume_from_checkpoint = True)


if IS_MAIN_PROCESS:
    model.save_pretrained_merged("chandomitra_phi4-4", tokenizer, save_method = "merged_16bit")


if __name__ == "__main__":
    print("Training completed successfully!")