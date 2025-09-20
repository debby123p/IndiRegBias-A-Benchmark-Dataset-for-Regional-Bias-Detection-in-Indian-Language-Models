# IndiRegBias-A-Benchmark-Dataset-for-Regional-Bias-Detection-in-Indian-Language-Models
## Overview
This research examines the effectiveness of Large Language Models (LLMs) in detecting regional bias within social media comments in the Indian context. In a culturally diverse nation like India, regional biases can perpetuate stereotypes and create social divisions. This project addresses the challenge that many current LLMs are trained primarily on Western datasets, leading to a lack of awareness of the nuanced biases prevalent in India.
Our goal is to make LLMs aware of regional bias in Indian content and improve their detection performance. This work contributes a novel dataset, a methodology for bias detection, and provides insights into the strengths and limitations of current LLMs for this task.

## Dataset

A core contribution of this research is a new, culturally-specific dataset of user-generated comments for detecting regional bias.

- Data Source: Over 200,000 comments were collected from Reddit and YouTube.

- Final Size: After cleaning, the final dataset consists of 25,000 user comments.
  
- Annotation: A rigorous human annotation process was implemented, involving two groups of three annotators. Inter-annotator agreement was high, with a Cohen's kappa value of 0.91 for binary classification and 0.83 for multi-class classification.

### Annotation Schema
Each of the 25,000 comments was manually classified using a multi-level severity schema:


Level 1: Bias Identification: Is the comment a regional bias (1) or not (0). 


Level 2: Severity of Bias: The severity of the comment was rated as Mild (1), Moderate (2), or Severe (3).


Level 3: State Identification: The name of the Indian state being targeted in the comment was recorded.

### Data Analysis Highlights

- Out of 25,000 comments,13,020 (52.1%) contained regional bias.
  ![image](https://github.com/user-attachments/assets/ba4fe26b-d17f-41d8-822b-9da3f3879e62)
  
- Severity breakdown of biased comments:


