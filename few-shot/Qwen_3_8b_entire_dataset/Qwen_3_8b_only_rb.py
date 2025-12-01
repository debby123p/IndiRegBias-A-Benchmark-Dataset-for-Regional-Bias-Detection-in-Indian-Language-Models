import os
import re
import gc
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
    "target_device": "cuda:1", # GPU ID
    "batch_size": 16, # Batch size
    "max_new_tokens": 256,
    "hf_token": "", # Hugging_face Token 
    "seed": 42
}
BASE_DIR = "" # Base directory to the data
OUTPUT_DIR = "" # Output directory
INPUT_DATA = os.path.join(BASE_DIR, "path_to_the_dataset") # Dataset file path
FEW_SHOT_SOURCE = [os.path.join(BASE_DIR, "path_to_the_support_examples")] # Output Directory

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

def get_model_and_tokenizer(model_id, device):
    """Load 4-bit quantized model and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=CONFIG["hf_token"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        token=CONFIG["hf_token"]
    )
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def build_few_shot_context(file_paths):
    """Construct few-shot examples string from CSV file."""
    examples = []
    for path in file_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                examples.append((str(row['comment']).strip(), int(row['level-1'])))
    
    random.seed(CONFIG["seed"])
    random.shuffle(examples)
    
    formatted_str = ""
    used_comments = set()
    
    for comment, label in examples:
        used_comments.add(comment)
        reasoning = "Contains regional bias." if label == 1 else "No regional bias."
        formatted_str += f"\n--- Example ---\nComment: \"{comment}\"\nReasoning: {reasoning}\nClassification: {label}\n--- End Example ---\n"
        
    return formatted_str, used_comments

def extract_prediction(response_text):
    """Parse model output for reasoning and classification label."""
    clean_text = re.sub(r'<.*?>', '', response_text, flags=re.DOTALL).strip()
    
    # Default values
    prediction = 0
    reasoning = clean_text.split("Classification:")[0].strip()
    
    # Regex for explicit classification
    match = re.search(r"Classification:\s*([01])", clean_text)
    if match:
        prediction = int(match.group(1))
    elif "REGIONAL BIAS" in clean_text.upper():
        prediction = 1
        
    return reasoning, prediction

def generate_report(csv_path):
    """Generate metrics and confusion matrix from results."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path).dropna(subset=['predicted_label'])
    y_true = df['true_label'].astype(int)
    y_pred = df['predicted_label'].astype(int)
    
    # Text Report
    report = classification_report(y_true, y_pred, target_names=['No Bias (0)', 'Bias (1)'], zero_division=0)
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    print(f"\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: 0', 'Pred: 1'],
                yticklabels=['True: 0', 'True: 1'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(MATRIX_IMG)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(CONFIG["target_device"] if torch.cuda.is_available() else "cpu")
    
    # Load resources
    model, tokenizer = get_model_and_tokenizer(CONFIG["model_id"], device)
    few_shot_prompt, exclude_comments = build_few_shot_context(FEW_SHOT_SOURCE)
    
    # Data Loading
    df_full = pd.read_csv(INPUT_DATA)
    processed_comments = set()
    if os.path.exists(PARTIAL_OUTPUT):
        df_prog = pd.read_csv(PARTIAL_OUTPUT)
        processed_comments = set(df_prog['comment'].astype(str).str.strip())
        print(f"Resuming from {len(processed_comments)} processed entries.")

    to_exclude = exclude_comments.union(processed_comments)
    df_process = df_full[~df_full['comment'].astype(str).str.strip().isin(to_exclude)].copy()
    
    print(f"Starting inference on {len(df_process)} samples.")

    batch_size = CONFIG["batch_size"]
    with torch.no_grad():
        for i in tqdm(range(0, len(df_process), batch_size), desc="Processing"):
            batch = df_process.iloc[i:i+batch_size]
            prompts = []
            
            # Prepare Prompts
            for _, row in batch.iterrows():
                user_content = f"{few_shot_prompt}\n--- Classify ---\nComment: \"{row['comment']}\""
                chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
                prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

            # Tokenize
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            try:
                # Generate
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
                    reasoning, pred = extract_prediction(text)
                    results.append({
                        'comment': batch.iloc[idx]['comment'],
                        'true_label': batch.iloc[idx]['level-1'],
                        'predicted_label': pred,
                        'reasoning': reasoning
                    })
                
                # Save Progress
                pd.DataFrame(results).to_csv(
                    PARTIAL_OUTPUT, 
                    mode='a', 
                    header=not os.path.exists(PARTIAL_OUTPUT), 
                    index=False
                )

            except Exception as e:
                print(f"Batch error: {e}")
            
            # Memory Cleanup
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

    # Finalise
    if os.path.exists(PARTIAL_OUTPUT):
        if os.path.exists(FINAL_OUTPUT):
            os.remove(FINAL_OUTPUT)
        os.rename(PARTIAL_OUTPUT, FINAL_OUTPUT)
        print(f"Inference complete. Results saved to {FINAL_OUTPUT}")
        generate_report(FINAL_OUTPUT)

if __name__ == "__main__":
    main()
