# Customer Review Classification

Customer review sentiment and category classification using DistilBERT.

## Overview

This project uses DistilBERT to classify customer reviews across two tasks:

- **Sentiment classification:** Negative, Neutral, Positive
- **Category classification:** Product Related, Staff + Billing, Suggestion

The project includes custom dataset preparation, text preprocessing, transformer based model training, evaluation, and inference.

## Dataset

The project uses a custom generated customer review dataset containing **1,026 reviews** with sentiment and category labels.

### Labels

**Sentiment**
- Negative
- Neutral
- Positive

**Category**
- Product Related
- Staff + Billing
- Suggestion

The dataset was split into **820 training samples** and **206 evaluation samples** for the original experiments.

## Methodology

The project uses `distilbert-base-uncased` for both classification tasks.

### Sentiment Classification

A DistilBERT sequence classification model was fine-tuned to classify reviews into three sentiment classes: `Negative` · `Neutral` · `Positive`.

### Category Classification

A separate DistilBERT sequence classification model was fine-tuned to classify reviews into: `Product Related` · `Staff + Billing` · `Suggestion`.

For both models, the original training setup used tokenization with a maximum sequence length of 512 tokens and a batch size of 8.

## Results

### Sentiment Classification

| Metric | Score |
|---|---:|
| Accuracy | 95.63% |
| Weighted Precision | 95.75% |
| Weighted Recall | 95.63% |
| Weighted F1-score | 95.64% |

The sentiment model achieved 95.63% accuracy on 206 evaluation samples.

### Category Classification

| Metric | Score |
|---|---:|
| Accuracy | 100.00% |
| Weighted Precision | 100.00% |
| Weighted Recall | 100.00% |
| Weighted F1-score | 100.00% |

The category model achieved 100% accuracy on the 206-sample evaluation split.

> **Evaluation note:** The original category training procedure evaluated this same split during training. Therefore, the 100% result should not be interpreted as performance on a completely untouched held-out test set.

## Inference

The trained models can be used to classify new customer reviews.

Example sentiment output:

```text
Input:  "The staff was very helpful and the service was quick."
Output: Positive
```

## Limitations

- The dataset is custom generated and relatively small compared with large-scale NLP datasets.
- The original experiments used an 80/20 train/evaluation split.
- For the category model, the evaluation split was also used for epoch-wise evaluation during training, so the reported 100% score should not be treated as an unbiased held-out test result.
- The project focuses on three predefined sentiment classes and three predefined review categories.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Pandas
- scikit-learn
- NumPy

## Project Structure

```text
customer-review-classification/
├── data/
├── models/
├── results/
├── src/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
