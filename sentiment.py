import pandas as pd
import torch

from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

train_df = pd.read_csv("sentiment_train.csv")
test_df = pd.read_csv("sentiment_test.csv")

label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

train_df["label"] = train_df["Sentiment"].map(label_map)
test_df["label"] = test_df["Sentiment"].map(label_map)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    train_df["Review"].tolist(),
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    test_df["Review"].tolist(),
    truncation=True,
    padding=True
)

print("Tokenization complete")


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



train_dataset = ReviewDataset(
    train_encodings,
    train_df["label"].tolist()
)

test_dataset = ReviewDataset(
    test_encodings,
    test_df["label"].tolist()
)

print("Datasets created")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label={
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    },
    label2id={
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
)

print("Model loaded")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10
)

print("Training arguments ready")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print("Trainer ready")

trainer.train()

print("Training complete")

predictions = trainer.predict(test_dataset)

preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)

print(f"Accuracy: {accuracy:.4f}")

model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

print("Model saved")