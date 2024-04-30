from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch

# Read the CSV file containing labeled data
df = pd.read_csv("query_classification_labeled.csv")

train_ratio = 0.8  # 80% for training, 20% for evaluation
train_size = int(train_ratio * len(df))
train_df = df[:train_size]
eval_df = df[train_size:]

# Load the pre-trained tokenizer and model
checkpoint = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

model.config.pad_token_id = tokenizer.pad_token_id

# Create a custom dataset class
class QueryClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.questions = df["Question"].tolist()
        self.labels = df["Label"].map({"SQL": 0, "Visualization": 1}).tolist()
        self.encodings = tokenizer(self.questions, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Create the training dataset
train_dataset = QueryClassificationDataset(train_df)

# Create the evaluation dataset
eval_dataset = QueryClassificationDataset(eval_df)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
print("Fine-tuned model saved.")
