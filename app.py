import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('./finetuned_industrial_model')
model = AutoModelForSequenceClassification.from_pretrained('./finetuned_industrial_model')

def predict_sentiment(report):
    inputs = tokenizer(report, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1).max().item()
    label = "✅ POSITIVE (Normal Operation)" if prediction == 1 else "🚨 NEGATIVE (Issue/Urgent)"
    return f"{label}\n\nConfidence: {confidence*100:.2f}%"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a maintenance/technician report...", label="Industrial Report"),
    outputs=gr.Textbox(label="Prediction"),
    title="🏭 Industrial Maintenance Sentiment Classifier",
    description="Fine-tuned DistilBERT model that classifies maintenance reports as Normal Operation or Issue/Urgent. Demonstrates domain adaptation of NLP models for industrial automation use cases.",
    examples=[
        ["The motor is running smoothly with no unusual vibrations detected."],
        ["Bearing temperature has exceeded safe limits and requires immediate attention."],
        ["Routine inspection completed, all systems operating within normal parameters."],
        ["Critical failure detected in the cooling system, production halted."],
        ["Slight fluctuation in voltage observed but within acceptable tolerance range."],
        ["Unexpected shutdown occurred during shift change, root cause still under investigation."],
        ["Replaced worn gasket during scheduled downtime, machine back online ahead of schedule."],
        ["Recurring fault code on PLC unit despite previous firmware patch."],
        ["Thermal imaging scan shows no hotspots across the panel array."],
        ["Operator reported intermittent stalling on the conveyor under heavy load."],
        ["Lubrication levels topped up, no abnormal wear detected on gear teeth."],
        ["Pressure sensor readings inconsistent, recommend recalibration before next run."],
        ["Annual safety audit completed with zero non-conformances."],
        ["Spike in current draw noticed on Line 3 motor, monitoring closely."],
        ["End-of-line testing confirms units pass torque and vibration specifications."],
    ]
)

demo.launch()