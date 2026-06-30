from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from industrial_data import industrial_reports
import numpy as np

# ✅ Step 1: Load tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ✅ Step 2: Split data (80% train, 20% validation)
split_idx = int(len(industrial_reports) * 0.8)
train_data = industrial_reports[:split_idx]
val_data = industrial_reports[split_idx:]

print(f"✅ Training samples: {len(train_data)}")
print(f"✅ Validation samples: {len(val_data)}")

# ✅ Step 3: Create PyTorch Dataset class
class IndustrialDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IndustrialDataset(train_data, tokenizer)
val_dataset = IndustrialDataset(val_data, tokenizer)

# ✅ Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=5,
    load_best_model_at_end=True,
)

# ✅ Step 5: Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# ✅ Step 6: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ✅ Step 7: Fine-tune!
print("\n✅ Starting fine-tuning...")
trainer.train()
print("✅ Fine-tuning complete!")

# ✅ Step 8: Save the fine-tuned model
model.save_pretrained('./finetuned_industrial_model')
tokenizer.save_pretrained('./finetuned_industrial_model')
print("✅ Fine-tuned model saved!")