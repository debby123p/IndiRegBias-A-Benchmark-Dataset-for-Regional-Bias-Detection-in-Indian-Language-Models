import os
import sys
import re
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
    "context_window": 8192,
    "inference_sample_size": 1000,
    "seed": 42
}


BASE_DIR = "" # Base directory to the data
OUTPUT_DIR = "" # Output directory

# Few-shot example
FEW_SHOT_SOURCES = [
    os.path.join(BASE_DIR, ""),
    os.path.join(BASE_DIR, "")
]

# Inference data source
INFERENCE_FILES = [
    os.path.join(BASE_DIR, "comments_for_inference_newest.csv")
]

# Saving the results
RESULTS_CSV = os.path.join(OUTPUT_DIR, "classification_results.csv")
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
    """Load 4-bit quantised model and tokenizer."""
    print(f"Loading 4-bit quantized model: {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
  
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

def compile_all_few_shot_examples(file_paths):
    """Aggregates examples from CSV files."""
    print("Compiling few-shot prompt from all provided files...")
    df_list = []
    
    for fpath in file_paths:
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                if 'level-1' in df.columns:
                    valid_df = df[df['level-1'].isin([0, 1])].copy()
                    df_list.append(valid_df)
                    print(f" - Loaded {len(valid_df)} examples from {os.path.basename(fpath)}")
                else:
                    print(f"Warning: Column 'level-1' missing in {os.path.basename(fpath)}")
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
        else:
            print(f"Warning: File not found: {fpath}")

    if not df_list:
        sys.exit("Error: No valid few-shot examples loaded.")

    
    df_combined = pd.concat(df_list, ignore_index=True)
    initial_count = len(df_combined)
    df_combined.drop_duplicates(subset=['comment'], inplace=True, keep='first')
    print(f"Consolidated {initial_count} examples into {len(df_combined)} unique examples.")

    
    formatted_prompt = ""
    used_comments = set()
    
    for _, row in df_combined.iterrows():
        comment = str(row['comment']).strip()
        label = int(row['level-1'])
        used_comments.add(comment)
        
        reasoning = "This example illustrates a comment with regional bias." if label == 1 else "This example illustrates a comment with no regional bias."
        formatted_prompt += f"\n--- Example ---\nComment: \"{comment}\"\nReasoning: {reasoning}\nClassification: {label}\n--- End Example ---\n"
        
    return formatted_prompt, used_comments

def extract_classification(response_text):
    """Parse reasoning and label from model output."""
    match = re.search(r"Classification:\s*([01])", response_text, re.IGNORECASE)
    prediction = int(match.group(1)) if match else -1
    
    reason_match = re.search(r"Reasoning:(.*?)(?=Classification:)", response_text, re.IGNORECASE | re.DOTALL)
    reasoning = reason_match.group(1).strip() if reason_match else "N/A"
    
    return reasoning, prediction

def generate_metrics(df):
    """Save classification report and confusion matrix."""
    valid_df = df[df['predicted_label'] != -1]
    
    if valid_df.empty:
        print("No valid predictions to report.")
        return

    y_true = valid_df['true_label'].astype(int)
    y_pred = valid_df['predicted_label'].astype(int)
    
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
    print(f"Using device: {device}")
    
    # Load resources
    model, tokenizer = setup_model(CONFIG["model_id"], CONFIG["device"])
  
    few_shot_prompt, exclude_comments = compile_all_few_shot_examples(FEW_SHOT_SOURCES)
    
    # Load Inference Data
    df_list = []
    print("\nLoading inference data...")
    for fpath in INFERENCE_FILES:
        if os.path.exists(fpath):
            df_list.append(pd.read_csv(fpath))
            print(f" - Loaded {len(df_list[-1])} comments from {os.path.basename(fpath)}")
        else:
            print(f"Warning: File not found: {fpath}")
            
    if not df_list:
        sys.exit("Error: No inference data loaded.")
        
    df_full = pd.concat(df_list, ignore_index=True)
    
    
    if len(df_full) > CONFIG["inference_sample_size"]:
        print(f"Dataset too large ({len(df_full)}). Sampling {CONFIG['inference_sample_size']} random entries.")
        df_full = df_full.sample(n=CONFIG["inference_sample_size"], random_state=CONFIG["seed"])
    else:
        print(f"Dataset size ({len(df_full)}) is within limits. Using full dataset.")

    # Filter out examples used in the prompt
    df_process = df_full[~df_full['comment'].astype(str).str.strip().isin(exclude_comments)].copy()
    print(f"Proceeding with {len(df_process)} unique comments for inference.")

    # Inference Loop
    results = []
    batch_size = CONFIG["batch_size"]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df_process), batch_size), desc="Inference"):
            batch = df_process.iloc[i:i+batch_size]
            prompts = []
            
            for _, row in batch.iterrows():
                user_msg = f"{few_shot_prompt}\n--- Classify the following comment ---\nComment: \"{row['comment']}\""
                chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
                prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=CONFIG["context_window"]
            ).to(device)
            prompt_len = inputs.input_ids.shape[1]
            max_new = max(100, CONFIG["context_window"] - prompt_len - 10)

            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            for idx, text in enumerate(decoded):
                reasoning, pred = extract_classification(text)
                results.append({
                    'comment': batch.iloc[idx]['comment'],
                    'true_label': batch.iloc[idx]['level-1'],
                    'predicted_label': pred,
                    'reasoning': reasoning
                })

    # Finalise
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Inference complete. Results saved to {RESULTS_CSV}")
    
    generate_metrics(results_df)

if __name__ == "__main__":
    main()
