from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained(
    "Jenifer2606/vstar-customer-category-distilbert"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "Jenifer2606/vstar-customer-category-distilbert"
)

df = pd.read_csv("category_test.csv")

id2label = {
    0: "Product Related",
    1: "Staff + Billing",
    2: "Suggestion"
}

df = pd.read_excel("vstar_test_final.xlsx")

predictions = []

for review in df["Review"]:

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    predictions.append(id2label[prediction])

df["Predicted Category"] = predictions
df.to_excel("category_predictions.xlsx", index=False)

print("Predictions saved to category_predictions.xlsx")





