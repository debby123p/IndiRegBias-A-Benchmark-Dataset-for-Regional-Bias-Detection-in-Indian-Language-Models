import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
import numpy as np
import gc
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

CONFIG = {
    "model_id": "mistralai/Mistral-7B-v0.3",
    "paths": {
        "main_dataset": "",  # Base directory to the data
        "test_set": "", # Test Set
        "validation_set": "", # Validation Set
        "output_dir": "" # Output directory
    },
    "hyperparameters": {
        "lr": 2e-5,
        "epochs": 10,
        "batch_size": 4,
        "max_seq_length": 512
    },
    "label_map": {0: "non-regional bias", 1: "regional bias"},
    "label_map_reverse": {"non-regional bias": 0, "regional bias": 1}
}

# Data Preparation

def prepare_datasets():
    """
    Loads datasets, excludes test/val data from the train set, and standardises formats.
    """
    print("--- Loading and Preparing Datasets ---")
    try:
        df_main = pd.read_csv(CONFIG["paths"]["main_dataset"])
        df_test = pd.read_csv(CONFIG["paths"]["test_set"])
        df_val = pd.read_csv(CONFIG["paths"]["validation_set"])
        
        # Ensure string types for matching
        for df in [df_main, df_test, df_val]:
            df['comment'] = df['comment'].astype(str)

        print(f"Raw Sizes -> Main: {len(df_main)}, Test: {len(df_test)}, Val: {len(df_val)}")

        # 1. Filter Training Data 
        exclude_comments = set(df_val['comment']).union(set(df_test['comment']))
        df_train = df_main[~df_main['comment'].isin(exclude_comments)].copy()
        df_train['level-1'] = df_train['level-1'].astype(int)

        # 2. Process Validation Data
        df_val = df_val[['comment', 'true_label']].copy()
        df_val['true_label'] = df_val['true_label'].astype(str)
        df_val['level-1'] = df_val['true_label'].map(CONFIG["label_map_reverse"])

        # 3. Process Test Data
        df_test = df_test[['comment', 'true_label']].copy()
        df_test['true_label'] = df_test['true_label'].astype(str)
        df_test['level-1'] = df_test['true_label'].map(CONFIG["label_map_reverse"])

        # Handle Mapping Errors
        for name, df in [("Validation", df_val), ("Test", df_test)]:
            if df['level-1'].isnull().any():
                print(f"Warning: Dropping rows with unknown labels in {name} set.")
                df.dropna(subset=['level-1'], inplace=True)
            df['level-1'] = df['level-1'].astype(int)

        print(f"Final Sizes -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        
        return {
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test)
        }

    except Exception as e:
        raise RuntimeError(f"Data preparation failed: {e}")

# Tokenizer & Formatting

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Custom Chat Template for Mistral [INST] format
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """
        {{ bos_token }}
        {% for message in messages %}
            {% if message['role'] == 'user' %}
                {{ '[INST] ' + message['content'] + ' [/INST]' }}
            {% elif message['role'] == 'assistant' %}
                {{ message['content'] + eos_token }}
            {% endif %}
        {% endfor %}
        """
    return tokenizer

def format_prompt(row, tokenizer):
    label = CONFIG["label_map"][row['level-1']]
    system_msg = "You are a helpful assistant trained to detect regional bias in text. Your task is to classify the user's comment as either 'regional bias' or 'non-regional bias'."
    user_msg = f"{system_msg}\nPlease classify the following comment: {row['comment']}"
    
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": label}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

# Training 

def run_training(datasets, tokenizer):
    print("\n--- Initializing Training ---")
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"], dtype=torch.bfloat16, device_map="cuda"
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
        output_dir=CONFIG["paths"]["output_dir"],
        num_train_epochs=CONFIG["hyperparameters"]["epochs"],
        per_device_train_batch_size=CONFIG["hyperparameters"]["batch_size"],
        gradient_accumulation_steps=2,
        optim="adamw_8bit",
        logging_steps=25,
        learning_rate=CONFIG["hyperparameters"]["lr"],
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataset_text_field="text",
        max_length=CONFIG["hyperparameters"]["max_seq_length"],
        packing=False,
    )

    # Format datasets for Trainer
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

    trainer.train()
    trainer.save_model(CONFIG["paths"]["output_dir"])
    print(f"Model saved to {CONFIG['paths']['output_dir']}")
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

# Evaluation

def evaluate_model(split_name, dataset, tokenizer):
    print(f"\n--- Starting Evaluation: {split_name.upper()} ---")
    
    # Reload Model
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"], dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, CONFIG["paths"]["output_dir"])
    model.eval()

    predictions = []
    true_labels = []

    for example in tqdm(dataset, desc=f"Evaluating {split_name}"):
        true_labels.append(example['true_label'])
        
        system_msg = "You are a helpful assistant trained to detect regional bias in text. Your task is to classify the user's comment as either 'regional bias' or 'non-regional bias'."
        user_msg = f"{system_msg}\nPlease classify the following comment: {example['comment']}"
        messages = [{"role": "user", "content": user_msg}]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, eos_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

        if "non-regional bias" in response:
            predictions.append("non-regional bias")
        elif "regional bias" in response:
            predictions.append("regional bias")
        else:
            predictions.append("unknown")

    # Metrics & Saving
    print(f"\n--- {split_name} Classification Report ---")
    print(classification_report(true_labels, predictions, labels=["regional bias", "non-regional bias"], zero_division=0))
    
    res_df = pd.DataFrame({
        'comment': [ex['comment'] for ex in dataset],
        'true_label': true_labels,
        'predicted_label': predictions
    })
    
    save_path = os.path.join(CONFIG["paths"]["output_dir"], f"{split_name.lower()}_results.csv")
    res_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

# Main Execution

if __name__ == "__main__":
    # 1. Setup
    datasets = prepare_datasets()
    tokenizer = get_tokenizer()

    # 2. Train
    run_training(datasets, tokenizer)

    # 3. Evaluate
    evaluate_model("Validation", datasets["validation"], tokenizer)
    evaluate_model("Test", datasets["test"], tokenizer)
    
    print("\nPipeline execution complete.")
