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
tokenizer = AutoTokenizer.from_pretrained("category_model")
model = AutoModelForSequenceClassification.from_pretrained("category_model")
model.eval()

# Load labeled test data
df = pd.read_csv("category_test.csv")

reviews = df["Review"].tolist()
true_labels = df["Category"].tolist()

label2id = {
    "Product Related": 0,
    "Staff + Billing": 1,
    "Suggestion": 2
}

id2label = {v: k for k, v in label2id.items()}

true_ids = [label2id[label] for label in true_labels]
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
accuracy = accuracy_score(true_ids, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_ids,
    predictions,
    average="weighted"
)

print("\n===== CATEGORY EVALUATION =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        true_ids,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "Product Related",
            "Staff + Billing",
            "Suggestion"
        ]
    )
)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(true_ids, predictions, labels=[0, 1, 2]))