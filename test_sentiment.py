from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "Jenifer2606/vstar-customer-sentiment-distilbert"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "Jenifer2606/vstar-customer-sentiment-distilbert"
)
id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

while True:
    review = input("\nEnter review: ")

    if review.lower() in ["exit", "quit"]:
        print("Goodbye")
        break

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    print("Sentiment:", id2label[prediction])
