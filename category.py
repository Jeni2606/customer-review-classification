import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

train_df = pd.read_csv("category_train.csv")
test_df = pd.read_csv("category_test.csv")

label_map = {
    "Product Related": 0,
    "Staff + Billing": 1,
    "Suggestion": 2
}

train_df["label"] = train_df["Category"].map(label_map)
test_df["label"] = test_df["Category"].map(label_map)

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
    0: "Product Related",
    1: "Staff + Billing",
    2: "Suggestion"
    },
    label2id={
    "Product Related": 0,
    "Staff + Billing": 1,
    "Suggestion": 2
    }
)

print("Model loaded")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    eval_strategy="epoch",      # Evaluate after every epoch
    save_strategy="epoch",      # Save checkpoint every epoch
    logging_strategy="epoch",   # Log metrics every epoch

    load_best_model_at_end=True
)

print("Training arguments ready")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Trainer ready")

trainer.train()
history = pd.DataFrame(trainer.state.log_history)
history.to_csv("training_history.csv", index=False)

print(history)

# Keep only rows that contain training loss
train_history = history[history["loss"].notna()][["epoch", "loss"]]

# Keep only rows that contain validation loss
eval_history = history[history["eval_loss"].notna()][["epoch", "eval_loss"]]

# Plot graph
plt.figure(figsize=(8,5))

plt.plot(
    train_history["epoch"],
    train_history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    eval_history["epoch"],
    eval_history["eval_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("Overfitting Analysis")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("overfitting_graph.png")
plt.show()

print("Training complete")

predictions = trainer.predict(test_dataset)

preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)

print(f"Accuracy: {accuracy:.4f}")

model.save_pretrained("category_model")
tokenizer.save_pretrained("category_model")

print("Model saved")