from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

# ✅ Step 1: Load a sentiment dataset (IMDB - we'll adapt it conceptually)
print("✅ Loading dataset...")
dataset = load_dataset("stanfordnlp/imdb")

print(f"✅ Train samples: {len(dataset['train'])}")
print(f"✅ Test samples: {len(dataset['test'])}")
print(f"\nSample review:\n{dataset['train'][0]['text'][:300]}")
print(f"Label: {dataset['train'][0]['label']}")  # 0 = negative, 1 = positive

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Step 2: Load pre-trained tokenizer and model
print("\n✅ Loading DistilBERT tokenizer and model...")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("✅ Model loaded successfully!")
print(f"Model labels: {model.config.id2label}")

# ✅ Step 3: Test on ABB-style maintenance reports
print("\n✅ Testing on industrial maintenance reports...\n")

test_reports = [
    "The motor is running smoothly with no unusual vibrations detected.",
    "Bearing temperature has exceeded safe limits and requires immediate attention.",
    "Routine inspection completed, all systems operating within normal parameters.",
    "Critical failure detected in the cooling system, production halted.",
    "The robotic arm calibration was successful and performance has improved significantly.",
    "Insulation damage found on multiple cables, urgent replacement needed."
]

for report in test_reports:
    inputs = tokenizer(report, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1).max().item()
    label = model.config.id2label[prediction]
    
    print(f"Report: {report}")
    print(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)\n")

    # ✅ Step 3.5: Save baseline results to file
with open('baseline_results.txt', 'w') as f:
    f.write("BASELINE MODEL RESULTS (Before Fine-tuning)\n")
    f.write("="*50 + "\n\n")
    for report in test_reports:
        inputs = tokenizer(report, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
        label = model.config.id2label[prediction]
        f.write(f"Report: {report}\n")
        f.write(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)\n\n")

print("✅ Baseline results saved to baseline_results.txt")