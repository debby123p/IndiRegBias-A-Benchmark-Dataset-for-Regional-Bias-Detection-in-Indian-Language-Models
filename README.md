# CAN LARGE LANGUAGE MODELS DETECT INDIAN REGIONAL BIAS? DATASET & ANALYSIS
## Overview
This research examines the effectiveness of Large Language Models (LLMs) in detecting regional bias within social media comments in the Indian context. In a culturally diverse nation like India, regional biases can perpetuate stereotypes and create social divisions. This project addresses the challenge that many current LLMs are trained primarily on Western datasets, leading to a lack of awareness of the nuanced biases prevalent in India.
Our goal is to make LLMs aware of regional bias in Indian content and improve their detection performance. This work contributes a novel dataset, a methodology for bias detection, and provides insights into the strengths and limitations of current LLMs for this task.

## Dataset

A key contribution of this research is a new, culturally specific dataset of user-generated comments for detecting regional bias.

- Data Source: Comments collected from Reddit and YouTube.

- Final Size: After cleaning, the final dataset consists of 25,000 user comments.
  
- Annotation: A rigorous human annotation process was implemented, involving two groups of three annotators. Inter-annotator agreement was high, with a Cohen's kappa value of 0.91 for binary classification and 0.83 for multi-class classification.

### Annotation Schema
Each of the 25,000 comments was manually classified using a multi-level severity schema:


Level 1: Bias Identification: Is the comment a regional bias (1) or not (0). 


Level 2: Severity of Bias: The severity of the comment was rated as Mild (1), Moderate (2), or Severe (3).


Level 3: State Identification: The name of the Indian state being targeted in the comment was recorded.

### Data Analysis Highlights

These comments are collected from videos or subreddit pages belonging to different regions, where the languages are mixed, such as English, Hinglish (a mix of Hindi and English), a mix of Bengali and English, a mix of Malayalam and English, or Marathi and English; thus, we have a multilingual and code-mixed language dataset.

- Out of 25,000 comments,13,020 (52.1%) contained regional biases, and 11,980 (47.9%) contained non-regional biases.
  
- Severity breakdown of biased comments is as Mild (3,747 comments), Moderate (6,663 comments), and Severe (2,610 comments).

- The data also showed the distribution of comments across different regions of India, with a large number of biased comments found for South India (5,802 comments) and North India (3,906 comments).

- In the state-wise breakdown, Kerala emerged as the most represented state with 1,841 comments, followed by Goa (1,103) and West Bengal (1,072).

## Methodology
The project's methodology is centred around a zero-shot classification experiment to evaluate the performance of various prominent LLMs.

- Zero-Shot Setting: This experimental setup starts with curating a prompt that follows the Chain-of-Thought prompting technique for the classification of comments into different class labels. The experiments under this were conducted using the 16-bit, that is, the full-precision models.
  
- Few-Shot Setting: Along with a prompt that follows the Chain-of-Thought prompting for the final inferencing task. The experiments under this were conducted using the 4-bit quantisation of the model.

- Fine-Tuning: This approach helps us to utilise our curated dataset on the base large language model by training it on our dataset. This helps to make the model gain expertise in recognising the task we are targeting, which is the classification of comments into regional bias or non-regional bias or understanding the severity levels of the comments. The following are the experimental techniques deployed:

  1) Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA)
Full fine-tuning of models with billions of parameters (e.g., Qwen3-8B) requires immense video memory (VRAM) and computational time to update all weights. To address this, we utilised Parameter-Efficient Fine-Tuning (PEFT). We implement Low-Rank Adaptation (LoRA), which is one of the techniques of Parameter-Efficient Fine-Tuning (PEFT).

     1) Experiment 1: Instruction-Based Supervised Fine-Tuning (SFT)
        
        For the binary classification task, we utilised instruction-based supervised fine-tuning(SFT). In this approach, data is formatted as a ”chat” or interaction, and we provide instructions to the model through               system messages for the classification of the comment by predicting the next token. Through the training phase, it goes through instruction-response pairs and then generates the write token for each comment in the         test set.
        
             • System Message: Sets the context (e.g., ”You are a helpful assistant... Your task is to classify...”).
        
             • User Message: Provides the specific input (e.g., ”Please classify the following comment: {comment}”).
        
             • Assistant Response: The model is trained to generate the ideal classification (e.g., ”Regional Bias”).
        
        The hyperparameters used for this experimental setup are detailed in the following table.
         ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(9).png)


     2) Experiment 2: Classification-Based Supervised Fine-Tuning (SFT)
        
        For the severity classification task (Mild, Moderate, Severe), we adopted a classification-based Supervised Fine-Tuning approach. The LLMs have a generative layer to predict the next token based on the                     instruction; this layer has been removed and replaced with a classification head. This new layer will robustly predict a number from 1 to 3 for the three class labels based on the instruction.

        The hyperparameters used for this experimental setup are detailed in the following table.
         ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(10).png)

### Models Evaluated
Eight prominent LLMs were selected for their instruction-following and reasoning abilities:

- Qwen/Qwen3-8B 

- Qwen/Qwen3-14B 

- mistralai/Mistral-7B-Instruct-v0.3 

- google/gemini-2.5-pro 

- meta-llama/Llama-3.2-3B 

- deepseek-ai/DeepSeek-R1-Distill-Llama-8B 

- google/gemma-1.1-7b-it 

- microsoft/Phi-4-mini-reasoning(4b) 

## Results

### Results: Binary Classification 

1) Zero-Shot Results

   The results for the models are presented in the zero-shot setting in the table.

   ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(11).png)

2) Few-Shot Results

   The few-shot experiment started with preparing the support with 500 examples (260 regional biases and 240 non-regional biases), with lower precision and F1-score values in comparison to the zero-shot score in both
   classes.

   ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(12).png)

      • The few-shot method required excessive memory (40-53 GB) and runtime (25-35 hours) even after using a 4-bit model.
   
      • Built-in safety features likely confused the identification of bias with the generation of it, defaulting to incorrect ”safe” answers.
   
      • 4-bit quantisation, necessary for the few-shot method, severely degraded the model’s reasoning and destroyed performance.

   We conducted the few-shot experiments with a smaller number of support (different combinations of regional biases and non-regional biases) on a smaller dataset of 1000 comments.

   ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(13).png)

   After looking through the inferences of the above experiment, we proceeded with experiments on the entire dataset with the type of support we have provided in Exp-2, Exp-3, and Exp-4, as we got improved performances in    comparison to the zero-shot results.

   ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(14).png)

   This few-shot strategy is a complete failure. In all the experiments, the Zero-Shot setting is significantly better and more balanced than the few-shot setting.

3) Fine-Tuning Results

   Fine-tuned Qwen emerged as the absolute best performer, achieving high reliability with F1-scores nearing 0.90 for both bias categories. Fine-tuned Mistral showed only minor improvements over its zero-shot (e.g., 0.61     to 0.65 F1), demonstrating far less adaptability to this specific task compared to Qwen.

   ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(15).png)

### Results: Multi-Class Classification

1) Zero-Shot Results

   The zero-shot model fails to distinguish between severity levels. It heavily over-predicts “Mild” cases while missing the vast majority of “Moderate” and “Severe” instances. This can be seen by the critically low
   F1-      scores of 0.20 and 0.15, respectively.

   ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(16).png)

3) Few-Shot Results

   As observed in the binary classification experiments, the few-shot performance was consistently poor in comparison to the zero-shot baseline. Consequently, we eliminated few-shot experiments for the multi-class            classification phase. This decision was driven by the significant imbalance in the dataset and the complexity required to curate varied support examples, which previous experiments showed led to model confusion rather     than improvement.

4) Fine-Tuning Results

   Needs to be updated.
   
## Conclusion

- Benchmark Creation: Developed a novel dataset to address the gap in regional bias detection resources.
  
- Prompting Analysis: Contrary to standard NLP trends, Few-Shot Learning proved detrimental, causing catastrophic forgetting and reduced generalisation compared to Zero-Shot settings.
  
- Fine-Tuning Efficacy: Utilised PEFT with LoRA, which significantly outperformed prompting strategies and effectively solved the binary bias detection task.
  
- Classification Challenges: While binary detection achieved high accuracy, Severity Classification remains a challenge due to the subtle nuances between severity levels.


## Future Work

- LLM vs. Human Annotation: Systematically compare LLM-generated annotations against our human-annotated gold standard to evaluate if LLMs can be trusted for this nuanced task. This is crucial for developing scalable methods for building large, culturally-aware datasets.

