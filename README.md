# IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models
## Overview
This research examines the effectiveness of Large Language Models (LLMs) in detecting regional bias within social media comments in the Indian context. In a culturally diverse nation like India, regional biases can perpetuate stereotypes and create social divisions. This project addresses the challenge that many current LLMs are trained primarily on Western datasets, leading to a lack of awareness of the nuanced biases prevalent in India.
Our goal is to make LLMs aware of regional bias in Indian content and improve their detection performance. This work contributes a novel dataset, a methodology for bias detection, and provides insights into the strengths and limitations of current LLMs for this task.

## Dataset

A key contribution of this research is a new, culturally specific dataset of user-generated comments for detecting regional bias.

- Data Source: Over 200,000 comments were collected from Reddit and YouTube.

- Final Size: After cleaning, the final dataset consists of 25,000 user comments.
  
- Annotation: A rigorous human annotation process was implemented, involving two groups of three annotators. Inter-annotator agreement was high, with a Cohen's kappa value of 0.91 for binary classification and 0.83 for multi-class classification.

### Annotation Schema
Each of the 25,000 comments was manually classified using a multi-level severity schema:


Level 1: Bias Identification: Is the comment a regional bias (1) or not (0). 


Level 2: Severity of Bias: The severity of the comment was rated as Mild (1), Moderate (2), or Severe (3).


Level 3: State Identification: The name of the Indian state being targeted in the comment was recorded.

### Data Analysis Highlights

These comments are collected from videos or subreddit pages belonging to different regions, where the languages are mixed, such as English, Hinglish, a mix of Bengali and English, a mix of Malayalam and English, or Marathi and English; thus, we have a multilingual and code-mixed language dataset.

- Out of 25,000 comments,13,020 (52.1%) contained regional bias.
  ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(3).png)
  
- Severity breakdown of biased comments:
  ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(4).png)

- Region-wise distribution
  ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(5).png)
  
- State-wise distribution 
  ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(6).png)


### High severity level for states and regions

The analysis of biased comments revealed several recurring negative themes and stereotypes targeting specific states:


- Crime and Corruption: States like Delhi and Uttar Pradesh were frequently stereotyped as hubs of crime and corruption.


- Social Intolerance: Haryana and Uttar Pradesh were often associated with social intolerance, specifically misogyny and casteism.


- Socio-Economic Failure: Bihar and Uttar Pradesh were commonly labelled as socio-economically "failed" or "backward".


The following is the distribution of the comments for different states and regions with higher percentages of severe comments:
 ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(7).png)

### Underrepresentation of Some States
The data showed a lower volume of comments related to certain smaller states. This disparity can be attributed to two primary factors:

- Geographic and Digital Barriers: Challenging terrain and lower population density in these regions can limit internet access, resulting in a smaller overall digital footprint.

- Marginalisation in National Discourse: The online narrative is often dominated by larger states, which can lead to the systematic sidelining of issues and voices from smaller regions.

The following is the distribution of the comments for different states and regions with the lowest number of  comments:
 ![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Binary%20Classification%20(8).png)

## Methodology
The project's methodology is centred around a zero-shot classification experiment to evaluate the performance of various prominent LLMs.



- Zero-Shot Classification: We used a zero-shot setting where models classify comments based solely on descriptive prompts, without any prior examples.



- Prompting: A uniform prompt was engineered to guide the models, employing a chain-of-thought to ask them to analyse comments for regional stereotypes.



- Evaluation: Model performance was systematically evaluated against the human-annotated ground truth using standard metrics: Precision, Recall, F1-Score, and Accuracy.

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
The models were run on the entire dataset of 25,000 comments. The performance for binary bias detection is summarised below
![image](https://github.com/debby123p/IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models/blob/main/Images/Mid-Sem%20Evaluation%20Presentation.png)

### Key Findings

- Top Performers: The Qwen models (14B and 8B) were the most effective, demonstrating high accuracy and a balanced ability to handle the task.

- Inherently Biased Models: Some models, like Google's Gemma and DeepSeek-R1, were found to be severely biased themselves. They were hypersensitive to identifying bias and almost incapable of recognising unbiased content.

- Weakest Models: Meta's Llama-3.2 showed weak performance, while Microsoft's Phi-4-mini was a "catastrophic failure," classifying nearly all content as non-regional.

## Conclusion
This research highlights a significant gap in the ability of current LLMs to understand regional biases within the Indian context, likely due to their Western-centric training data. While some models like Qwen show promise, the overall results indicate that zero-shot classification is not a turnkey solution. The novel 25,000-comment dataset created in this project will serve as a crucial benchmark for future research in this area.

## Future Work
The next steps for this project will focus on the following areas:


- Few-Shot Learning and Fine-Tuning: Explore more efficient methods for few-shot learning and fine-tuning to overcome previous computational challenges without compromising model performance.


- Multi-Class Classification: Expand experiments beyond binary classification to detect the severity of bias (mild, moderate, severe) and the specific states being targeted.


- LLM vs. Human Annotation: Systematically compare LLM-generated annotations against our human-annotated gold standard to evaluate if LLMs can be trusted for this nuanced task. This is crucial for developing scalable methods for building large, culturally-aware datasets.

