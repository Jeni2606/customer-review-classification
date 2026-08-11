# Customer Review Classification

Customer review sentiment and category classification using DistilBERT.

## Overview

This project uses DistilBERT to classify customer reviews across two tasks:

- **Sentiment classification:** Negative, Neutral, Positive
- **Category classification:** Product Related, Staff + Billing, Suggestion

The project includes custom dataset preparation, text preprocessing, transformer-based model training, evaluation, and inference.

## Dataset

The project uses a custom-generated customer review dataset containing **1,026 reviews** with sentiment and category labels.

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

A DistilBERT sequence classification model was fine-tuned to classify reviews into three sentiment classes:

- Negative
- Neutral
- Positive

### Category Classification

A separate DistilBERT sequence classification model was fine-tuned to classify reviews into three categories:

- Product Related
- Staff + Billing
- Suggestion

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

## Live Demo

A browser-based demo is deployed on Hugging Face Spaces using Transformers.js, so predictions run entirely client-side — no server, no API calls, no cost.

**[Launch the demo →](https://huggingface.co/spaces/Jenifer2606/_Classification_Demo_)**

Enter any customer review and get real-time sentiment (Negative / Neutral / Positive) and category (Product Related / Staff + Billing / Suggestion) predictions, powered by the same DistilBERT models trained in this repo.

## Model Hosting

Trained model weights are too large for GitHub, so they're hosted on Hugging Face:

| Model | Format | Repository |
|---|---|---|
| Sentiment classifier | PyTorch | [Jenifer2606/vstar-customer-sentiment-distilbert](https://huggingface.co/Jenifer2606/vstar-customer-sentiment-distilbert) |
| Category classifier | PyTorch | [Jenifer2606/vstar-customer-category-distilbert](https://huggingface.co/Jenifer2606/vstar-customer-category-distilbert) |
| Sentiment classifier | ONNX (for browser inference) | [Jenifer2606/vstar-customer-sentiment-distilbert-ONNX](https://huggingface.co/Jenifer2606/vstar-customer-sentiment-distilbert-ONNX) |
| Category classifier | ONNX (for browser inference) | [Jenifer2606/vstar-customer-category-distilbert-ONNX](https://huggingface.co/Jenifer2606/vstar-customer-category-distilbert-ONNX) |

## Local Inference

For command-line inference instead of the web demo:

\`\`\`bash
python test_sentiment.py   # interactive sentiment classification
python test_category.py    # category prediction, saves to category_predictions.xlsx
\`\`\`

Both scripts download the trained models automatically from Hugging Face on first run.

### Category Classification

Run:

```bash
python test_category.py
```

The script loads the trained category model from:

`Jenifer2606/vstar-customer-category-distilbert`

The model predicts one of the following categories:

- Product Related
- Staff + Billing
- Suggestion

## Limitations

- The dataset is custom-generated and relatively small compared with large-scale NLP datasets.
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
│   ├── category_train.csv
│   ├── category_test.csv
│   ├── sentiment_train.csv
│   ├── sentiment_test.csv
│   ├── README.md
│   └── models/
│       ├── README.md
│       └── results/
│           ├── README.md
│           ├── sentiment_results.txt
│           └── category_results.txt
│
├── category.py
├── sentiment.py
├── test_category.py
├── test_sentiment.py
├── evaluate_category.py
├── evaluate_sentiment.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Model Hosting

The trained models are hosted on Hugging Face:

- `Jenifer2606/vstar-customer-sentiment-distilbert`
- `Jenifer2606/vstar-customer-category-distilbert`

The large model weights are not stored directly in this GitHub repository. The inference scripts automatically download the required models from Hugging Face when they are run.
