import os
import re
import gc
import sys
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CONFIG = {
    "model_id": "Qwen/Qwen3-8B", # Model ID
    "device": "cuda:1", # GPU ID
    "batch_size": 16, # Batch size
    "max_new_tokens": 256,
    "hf_token": "", # Hugging_face Token 
    "seed": 42
}

BASE_DIR = "" # Base directory to the data
OUTPUT_DIR = "" # Output directory
INPUT_DATA = os.path.join(BASE_DIR, "") # Dataset file path
FEW_SHOT_SOURCES = [
    os.path.join(BASE_DIR, "path_to_the_support_examples"),
    os.path.join(BASE_DIR, "path_to_the_support_examples")
] # Support file path

# Saving the results
PARTIAL_OUTPUT = os.path.join(OUTPUT_DIR, "classification_results_in_progress.csv")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "classification_results.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "classification_report.txt")
MATRIX_IMG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")


SYSTEM_PROMPT = """
You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

Task: Classify the given comment as either "REGIONAL BIAS" (1) or "NON-REGIONAL BIAS" (0).

Definition:
- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, or their people based on linguistic, cultural, economic, political, or infrastructural aspects. The comments can reflect either positive or negative biases towards specific states or regions.
- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or unrelated to regional characteristics.

Step-by-Step Analysis Process:
Step 1: Identify Regional References
Think: Does this comment mention or refer to:
- Specific Indian states (e.g., Bihar, Kerala, Punjab, etc.)
- Regional groups (e.g., South Indians, North Indians, Biharis, etc.)
- Cities or regions within India
- Language communities within India

Step 2: Check for Elements reinforcing biases
Look for these patterns:
- Generalisations about people from a region ("All X are Y")
- Assumptions about regional characteristics
- Comparative statements that imply superiority/inferiority
- Overgeneralized cultural, linguistic, economic, political, or infrastructural claims
- Reinforcement of common regional stereotypes

Step 3: Assess the Nature of the Statement
Consider:
- Is this a factual observation or a generalised assumption?
- Does it reinforce existing biases or challenge them?
- Is it based on personal experience or broad generalisation?
- Does it promote understanding or perpetuate divisions?

Step 4: Final Classification
Based on the analysis above, classify as:
- REGIONAL BIAS (1) if the comment reinforces regional biases or stereotypes
- NON-REGIONAL BIAS (0) if the comment is neutral, factual, or doesn't contain regional bias.

Output Format:
Provide a brief reasoning followed by the classification.
Format: "Reasoning: [text] ... Classification: [0 or 1]"
"""

def setup_model(model_id, target_device):
    """Load model with 4-bit quantisation and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # trust_remote_code=True is required for Qwen architecture
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=target_device,
        trust_remote_code=True
    )
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def prepare_few_shot_data(file_paths):
    """Construct few-shot examples string from CSV file."""
    examples = []
    for path in file_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                examples.append((str(row['comment']).strip(), int(row['level-1'])))
    
    random.seed(CONFIG["seed"])
    random.shuffle(examples)
    
    formatted_prompt = ""
    used_comments = set()
    
    for comment, label in examples:
        used_comments.add(comment)
        reasoning = "This is an example of a comment with regional bias." if label == 1 else "This is an example of a comment with no regional bias."
        formatted_prompt += f"\n--- Example ---\nComment: \"{comment}\"\nReasoning: {reasoning}\nClassification: {label}\n--- End Example ---\n"
        
    return formatted_prompt, used_comments

def extract_classification(response_text):
    """Parse model output for reasoning and classification label."""
    clean_text = re.sub(r'<.*?>', '', response_text, flags=re.DOTALL).strip()
    reasoning = clean_text.split("Classification:")[0].strip() or "N/A"
    
    # Regex extraction for label
    match = re.search(r"Classification:\s*([01])", clean_text)
    if match:
        prediction = int(match.group(1))
    else:
        # Fallback keyword search
        if "REGIONAL BIAS" in clean_text.upper():
            prediction = 1
        elif "NON-REGIONAL BIAS" in clean_text.upper():
            prediction = 0
        else:
            prediction = 0 # Default fallback
            
    return reasoning, prediction

def generate_metrics(csv_path):
    """Save classification report and confusion matrix."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path).dropna(subset=['predicted_label'])
    y_true = df['true_label'].astype(int)
    y_pred = df['predicted_label'].astype(int)
    
    # Text Report
    report = classification_report(y_true, y_pred, target_names=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'], zero_division=0)
    with open(REPORT_FILE, 'w') as f:
        f.write("Classification Report\n=====================\n\n" + report)
    print(f"\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: 0', 'Pred: 1'],
                yticklabels=['Actual: 0', 'Actual: 1'])
    plt.title('Confusion Matrix')
    plt.savefig(MATRIX_IMG)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    
    # Load resources
    model, tokenizer = setup_model(CONFIG["model_id"], CONFIG["device"])
    few_shot_prompt, exclude_comments = prepare_few_shot_data(FEW_SHOT_SOURCES)
    
    # Data Loading
    df_full = pd.read_csv(INPUT_DATA)
    processed_comments = set()
    if os.path.exists(PARTIAL_OUTPUT):
        df_prog = pd.read_csv(PARTIAL_OUTPUT)
        processed_comments = set(df_prog['comment'].astype(str).str.strip())
        print(f"Resuming process. Skipping {len(processed_comments)} entries.")
    to_exclude = exclude_comments.union(processed_comments)
    df_process = df_full[~df_full['comment'].astype(str).str.strip().isin(to_exclude)].copy()
    
    print(f"Processing remaining {len(df_process)} comments...")
    
    
    batch_size = CONFIG["batch_size"]
    with torch.no_grad():
        for i in tqdm(range(0, len(df_process), batch_size), desc="Inference"):
            batch = df_process.iloc[i:i+batch_size]
            prompts = []

            # Prepare Prompt
            for _, row in batch.iterrows():
                user_msg = f"{few_shot_prompt}\n--- Classify the following comment ---\nComment: \"{row['comment']}\""
                chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
                prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
          
            # Tokenize
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

            try:
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

                # Parse Results
                results = []
                for idx, text in enumerate(decoded):
                    reasoning, pred = extract_classification(text)
                    results.append({
                        'comment': batch.iloc[idx]['comment'],
                        'true_label': batch.iloc[idx]['level-1'],
                        'predicted_label': pred,
                        'reasoning': reasoning
                    })
                
                # Parse Results
                pd.DataFrame(results).to_csv(
                    PARTIAL_OUTPUT, 
                    mode='a', 
                    header=not os.path.exists(PARTIAL_OUTPUT), 
                    index=False
                )

            except Exception as e:
                print(f"Batch Error: {e}")
            
            # Memory Cleanup
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

    # Finalise
    if os.path.exists(PARTIAL_OUTPUT):
        if os.path.exists(FINAL_OUTPUT):
            os.remove(FINAL_OUTPUT)
        os.rename(PARTIAL_OUTPUT, FINAL_OUTPUT)
        print(f"Inference complete. Results at: {FINAL_OUTPUT}")
        generate_metrics(FINAL_OUTPUT)

if __name__ == "__main__":
    main()
