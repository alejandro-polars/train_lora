"""
Fine-Tuning & Quantization Pipeline for GNC/Aerospace Context Alignment
Optimized for resource-constrained hardware using Unsloth & QLoRA.
"""

import torch
from unsloth import FastLanguageModel, get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Hardware & Model Architecture Optimization
max_seq_length = 2048  # Context window scaling for hardware memory constraints
dtype = None           # Auto-detect compute capability (float16 or bfloat16)
load_in_4bit = True    # 4-bit base model quantization to optimize VRAM allocations

print("[INFO] Loading pre-quantized base architecture (Llama-3-8B-Instruct)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 2. Parameter-Efficient Fine-Tuning (PEFT/LoRA) Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,               # Rank dimension for low-rank matrix adaptation
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory-saving activation checkpointing
    random_state=3407,
)

# 3. Chat Template Mapping & Conversational Formatting
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # Enforces native role structural mappings
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# 4. Dataset Ingestion & Tokenization
print("[INFO] Loading and processing local conversational dataset...")
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Supervised Fine-Tuning (SFT) Parameters
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Simulates higher batch sizes with minimal VRAM
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)

# 6. Execution Loop
print("--- STARTING TRAINING PIPELINE ---")
trainer.train()
print("--- TRAINING SUCCESSFULLY COMPLETE ---")

# 7. Low-Latency Edge Deployment Compilation (.GGUF)
print("[INFO] Compiling model weights into standalone GGUF format...")
model.save_pretrained_gguf("model_final_q4", tokenizer, quantization_method="q4_k_m")
print("--- COMPILED BINARY GENERATED SUCCESSFULY ---")
