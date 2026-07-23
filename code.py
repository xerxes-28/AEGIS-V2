#install Dependencies and Login

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<=0.24.0" peft accelerate bitsandbytes
!pip install "datasets<4.4.0"
!pip install huggingface_hub

from huggingface_hub import login
login()

!pip install wandb
from Wandb import login
wandb.login()

# Load The model
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer 
from transformers import TrainingArguments
import json

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
    max_seq_length=8192,
    load_in_4bit=True,
)

# Load the LoRA 
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=True,
)

# import the dataset
import os
from google.colab import drive
from datasets import load_dataset, concatenate_datasets, Dataset
drive.mount('/content/drive')

sft_dataset = load_dataset("json", data_files="/content/drive/MyDrive/aegis_sft_dataset.jsonl", split="train")

def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

sft_dataset = sft_dataset.map(formatting_prompts_func, batched=True)

...

your_data = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/A.E.G.I.S-V.1/Combined-Data.json",
    split="train"
)
print(f"✅ Your dataset:        {len(your_data)}")

# merge all datasets into one and shuffle
combined = concatenate_datasets([
    keep_cols(your_data),
]).shuffle(seed=42)

print(f"\n🔢 Total examples: {len(combined)}")

dataset = combined.map(format_prompts, batched=True)
print("✅ Dataset formatted")

# Save to Drive so resumed sessions skip this section
DATASET_PATH = "/content/drive/MyDrive/./."
os.makedirs(DATASET_PATH, exist_ok=True)
dataset.save_to_disk(DATASET_PATH)
print(f"✅ Dataset saved → {DATASET_PATH}")


# SECTION 5 — LOAD SAVED DATASET
# from datasets import load_from_disk
# dataset = load_from_disk(
#     "/content/drive/MyDrive/A.E.G.I.S-V.1/formatted_dataset"
# )
# print(f"✅ Dataset loaded: {len(dataset)} examples")

# SECTION 6 — SFT TRAINING

import os
import torch
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

CURRENT_LR = 2e-5

training_args = SFTConfig(
    dataset_text_field="text",
    max_seq_length=8192,   
    packing=True,           
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,     
    max_steps=...,
    learning_rate=CURRENT_LR,
    warmup_steps=150,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",  
    weight_decay=0.01,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    output_dir="/content/drive/MyDrive/embedded-coder-checkpoints",
    report_to="wandb",
    run_name="AEGIS-V2.0",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    args=SFTConfig(
        dataset_text_field="text",
        **Training_CONFIG
    ),
)

#CHECKPOINT = "/content/drive/MyDrive/embedded-coder-checkpoints/."

# Section 7 - DOP Training -1
import json
import os
from google.colab import drive
from datasets import load_dataset, concatenate_datasets, Dataset
drive.mount('/content/drive')

dop_dataset = load_dataset("json", data_files="/content/drive/MyDrive/.", split="train")

def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

sft_dataset = sft_dataset.map(formatting_prompts_func, batched=True)

...

your_data = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/./.",
    split="train"
)
print(f"✅ Your dataset:        {len(your_data)}")

 dataset = load_from_disk(
    "/content/drive/MyDrive/./." )

# DOP Training -2
from unsloth import FastLanguageModel,PatchDOPTrainer
from trl import DOPTrainer, DOPConfig

patchDOPTrainer = PatchDOPTrainer()

model, tokenizer = FastLanguageModel.from_pretrained(
   "./content/drive/MyDrive/./.",  # Save the SFT model then Load it here for DOP training
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)

dpo_dataset = load_dataset("json", data_files="/content/drive/MyDrive/./.", split="train")

trainer = DOPTrainer(
   model=model,
   ref_model=None,
   tokenizer=tokenizer,
    train_dataset=dpo_dataset["train"],
    args=DOPConfig(
       learning_rate=5e-5,
       beta=0.1,
       per_device_train_batch_size=1,

         gradient_accumulation_steps=16,
         num_train_epochs=1,
         bf16=True,
         output_dir="/content/drive/MyDrive/./.",
    ),
)

trainer.train()

