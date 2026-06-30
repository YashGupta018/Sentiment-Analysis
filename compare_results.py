from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Load fine-tuned model
print("✅ Loading fine-tuned model...\n")
tokenizer = AutoTokenizer.from_pretrained('./finetuned_industrial_model')
model = AutoModelForSequenceClassification.from_pretrained('./finetuned_industrial_model')

test_reports = [
    "The motor is running smoothly with no unusual vibrations detected.",
    "Bearing temperature has exceeded safe limits and requires immediate attention.",
    "Routine inspection completed, all systems operating within normal parameters.",
    "Critical failure detected in the cooling system, production halted.",
    "The robotic arm calibration was successful and performance has improved significantly.",
    "Insulation damage found on multiple cables, urgent replacement needed."
]

print("AFTER FINE-TUNING RESULTS")
print("="*50 + "\n")

with open('finetuned_results.txt', 'w') as f:
    f.write("FINE-TUNED MODEL RESULTS (After Fine-tuning on Industrial Data)\n")
    f.write("="*50 + "\n\n")
    for report in test_reports:
        inputs = tokenizer(report, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        
        print(f"Report: {report}")
        print(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)\n")
        
        f.write(f"Report: {report}\n")
        f.write(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)\n\n")

print("✅ Fine-tuned results saved to finetuned_results.txt")