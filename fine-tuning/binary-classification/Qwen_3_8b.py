import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback, 
)
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import gc


CONFIG = {
    "model_id": "Qwen/Qwen3-8B", # Model ID
    "dataset_path": "", # Dataset file path
    "output_dir": "", # Output directory
    "seed": 42,
    "max_seq_length": 512,
    "label_map": {0: "non-regional bias", 1: "regional bias"}
}


def load_and_split_data(csv_path, seed=42):
    """Loads CSV, ensures types, and performs Train/Val/Test splits."""
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df['level-1'] = df['level-1'].astype(int)
        df['comment'] = df['comment'].astype(str)
        dataset = Dataset.from_pandas(df)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Split: 70% Train, 30% Temp
    train_val_split = dataset.train_test_split(test_size=0.3, seed=seed)
    # Split Temp: 10% Val, 20% Test 
    val_test_split = train_val_split['test'].train_test_split(test_size=(2/3), seed=seed)

    datasets = {
        "train": train_val_split['train'],
        "validation": val_test_split['train'],
        "test": val_test_split['test']
    }
    
    print(f"Data Split -> Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}, Test: {len(datasets['test'])}")
    return datasets

def format_prompt(row, tokenizer):
    """Formats a row into the Qwen chat template."""
    label = CONFIG["label_map"][row['level-1']]
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant trained to detect regional bias in text. Your task is to classify the user's comment as either 'regional bias' or 'non-regional bias'."
        },
        {"role": "user", "content": f"Please classify the following comment: {row['comment']}"},
        {"role": "assistant", "content": label}
    ]
    
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Training Function

def run_training(datasets, tokenizer):
    print("--- Initializing Model for Training ---")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Training Arguments
    training_args = SFTConfig(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="adamw_8bit",
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        
        # Evaluation & Saving
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=False,
    )

    # Apply formatting for Trainer
    train_data = datasets["train"].map(lambda x: format_prompt(x, tokenizer), num_proc=4)
    val_data = datasets["validation"].map(lambda x: format_prompt(x, tokenizer), num_proc=4)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )

    print("--- Starting Training ---")
    trainer.train()
    
    print(f"--- Saving Best Model to {CONFIG['output_dir']} ---")
    trainer.save_model(CONFIG["output_dir"])
    
    # Cleanup memory
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

# Evaluation Function

def evaluate_model(split_name, dataset, tokenizer):
    print(f"\n--- Starting Evaluation on: {split_name.upper()} ---")
    
    # Reload Base Model + Adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, CONFIG["output_dir"])
    model.eval()

    predictions = []
    true_labels = []

    for example in tqdm(dataset, desc=f"Evaluating {split_name}"):
        true_labels.append(CONFIG["label_map"][example['level-1']])
        
        # Construct Inference Prompt
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant trained to detect regional bias in text. Your task is to classify the user's comment as either 'regional bias' or 'non-regional bias'."
            },
            {"role": "user", "content": f"Please classify the following comment: {example['comment']}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

        if "non-regional bias" in response:
            predictions.append("non-regional bias")
        elif "regional bias" in response:
            predictions.append("regional bias")
        else:
            predictions.append("unknown")

    # Metrics
    print(f"\n--- {split_name.title()} Classification Report ---")
    print(classification_report(true_labels, predictions, labels=["regional bias", "non-regional bias"]))
    
    # Save Results
    res_df = pd.DataFrame({
        'comment': [ex['comment'] for ex in dataset],
        'true_label': true_labels,
        'predicted_label': predictions
    })
    save_path = os.path.join(CONFIG["output_dir"], f"{split_name}_results.csv")
    res_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    # Cleanup
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

# Main Execution 

if __name__ == "__main__":
    # 1. Prepare Data
    raw_datasets = load_and_split_data(CONFIG["dataset_path"], seed=CONFIG["seed"])
    tokenizer = get_tokenizer(CONFIG["model_id"])

    # 2. Train
    run_training(raw_datasets, tokenizer)

    # 3. Evaluate Validation Set
    evaluate_model("validation", raw_datasets["validation"], tokenizer)

    # 4. Evaluate Test Set
    evaluate_model("test", raw_datasets["test"], tokenizer)
    
    print("\nAll pipeline steps completed successfully.")
