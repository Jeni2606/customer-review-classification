from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import pandas as pd
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
model.eval()

# Load test data
df = pd.read_csv("sentiment_test.csv")

reviews = df["Review"].tolist()
true_labels = df["label"].tolist()

predictions = []

for review in reviews:
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    predictions.append(prediction)

# Metrics
accuracy = accuracy_score(true_labels, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="weighted"
)

print("\n===== SENTIMENT EVALUATION =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        true_labels,
        predictions,
        target_names=["Negative", "Neutral", "Positive"]
    )
)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(true_labels, predictions))