# AI-Powered Tool for Identifying and Supporting At-Risk Youth in Crisis Situations

## Project Overview

This project implements a multi-task model that outputs a label, urgency, and distress level from user text. These outputs are fed into a Q&A model along with the original text to generate tailored recommendations in real time. 
**Note:**
- The text input is given the highest weight during training.
- Data was scraped from Reddit posts and preprocessed for BART.
- BART provided sentiment scores that were used to hardcode urgency (immediate, urgent, non-urgent), overall sentiment(negative, positive, neutral), and distress level (low, med, high).
- Based on these scores, a generic response is generated (for example, advising immediate medical assistance and to call 911 if urgency is high).
- The original data along with these scores is now fed into a Groq LLM via prompt engineering, replacing the earlier GPT-2 fine-tuning approach, to generate therapy responses.
- There is ongoing consideration regarding the impact of sending urgency, overall sentiment, and distress level to the Groq LLM via prompt engineering.

## Introduction

The tool aims to identify distress signs in youth by analyzing text input using Natural Language Processing (NLP) and machine learning. It provides actionable insights and support, including a dedicated Q&A feature that further assists users in crisis situations.

## 1. Data Collection and Preparation

### Collect Diverse Data
- Expand the dataset to include various crisis-related text (e.g., self-harm, suicidal ideation, stress, anxiety).

### Preprocessing
- **Text Cleaning:** Remove stop words, punctuation, etc.
- **Tokenization & Lemmatization:** Break text into tokens and normalize them.
- **Handling Imbalanced Data:** Apply techniques like oversampling or data augmentation.

## 2. Feature Engineering

- **POS Tagging:** Extract parts-of-speech for syntactic insights.
- **Named Entity Recognition (NER):** Identify key entities in the text.
- **Sentence Length & Complexity Metrics:** Assess sentence structure and complexity.
- **Sentiment Analysis:** Use models to derive sentiment scores.
- **Word Embeddings:** Utilize pre-trained embeddings (Word2Vec, GloVe, BERT) for contextual understanding.

### Additional Features
- Linguistic features (POS tags, named entities, sentence length, complexity).
- Derived sentiment scores to serve as additional features.


## 3. Sentiment Classification

### Model Training
- Train a sentiment classifier (e.g., using BERT) to detect positive, neutral, or negative sentiments.

### Fine-Tuning
- Fine-tune on crisis-specific data to better capture distress-related text nuances.

## 4. Crisis Detection and Severity Analysis

### Risk Level Classification
- Develop a model to classify the severity of a crisis (e.g., low, moderate, high).

### Multi-Class Classification
- Implement classifiers to detect specific crisis types (e.g., bullying, suicidal ideation).

### Model Features
- **Crisis Type Classification:** Identify types like bullying or suicidal thoughts.
- **Urgency Detection:** Classify urgency levels (immediate, urgent, non-urgent).
- **Distress Level Classification:** Determine distress levels (low, moderate, high).

## 6. Q&A Feature Modeling

### Question Understanding
- Use intent detection and entity recognition to accurately interpret user queries.

### Response Generation
- Implement generative models (e.g., using Groq LLM’s prompt engineering) or retrieval-based systems to generate appropriate responses.

### Personalized Feedback
- Leverage user data to provide customized recommendations and feedback.

## 7. Evaluation and Fine-Tuning

### Model Evaluation
- Evaluate models using metrics such as accuracy, F1-score, precision, and recall.

### User Feedback
- Integrate user feedback for iterative improvements and model adjustments.

### Continuous Learning
- Develop mechanisms for continuous learning and periodic model updates based on new data.

## 8. Ethical Considerations

- **Bias Mitigation:** Ensure models do not perpetuate biases or provide harmful recommendations.
- **Explainability:** Use techniques like LIME or SHAP to make model decisions transparent to users and professionals.

## 9. Integration with Platform

### Real-Time Analysis
- Deploy models in a real-time pipeline to provide immediate feedback.

### Historical Analysis
- Use historical data to continuously improve model performance over time.

## Future Prospects

- **Social Group Recommendations:** Integrate the ability to recommend online support groups or communities based on sentiment analysis.
- **Advanced Social Network Analysis:** Apply NLP and graph-based analysis to map user connections and better match users with support resources, for example Therapists, Counselors or psychologists near their area.
