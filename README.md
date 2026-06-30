# Sentiment-Analysis — Industrial Maintenance NLP (Hugging Face)

Its a NLP project that fine-tunes a pre-trained transformer model (DistilBERT) to classify industrial maintenance and technician reports as **Positive** (normal operation) or **Negative** (issue/urgent), demonstrating domain adaptation — a core real-world NLP challenge in industrial automation contexts.

---

## Project Overview

General-purpose sentiment models (trained on movie reviews, product reviews, etc.) often fail when applied to specialized domains like industrial maintenance logs, because words like "vibrations," "parameters," and "bearing" carry different connotations than everyday language. This project demonstrates that gap directly, then fixes it through fine-tuning on a small, domain-specific labeled dataset.

The result is a measurable before/after case study: a baseline pre-trained model misclassifying maintenance reports, followed by a fine-tuned version that correctly classifies all of them.

---

## Problem Demonstrated: Domain Gap

The baseline model used is `distilbert-base-uncased-finetuned-sst-2-english`, pre-trained on the SST-2 movie review sentiment dataset. When tested on six industrial maintenance report sentences, it misclassified 2 out of 6 — including reports that were clearly positive (e.g., "running smoothly," "operating within normal parameters") — because it had never seen this kind of language during training.

### Baseline Results (Before Fine-tuning)

| Report | Prediction | Correct? |
|---|---|---|
| The motor is running smoothly with no unusual vibrations detected. | NEGATIVE | ❌ |
| Bearing temperature has exceeded safe limits and requires immediate attention. | NEGATIVE | ✅ |
| Routine inspection completed, all systems operating within normal parameters. | NEGATIVE | ❌ |
| Critical failure detected in the cooling system, production halted. | NEGATIVE | ✅ |
| The robotic arm calibration was successful and performance has improved significantly. | POSITIVE | ✅ |
| Insulation damage found on multiple cables, urgent replacement needed. | NEGATIVE | ✅ |

**Baseline accuracy on industrial text: 4/6 (66.7%)**

---

## Fix: Fine-tuning on Domain-Specific Data

A small labeled dataset of 30 industrial maintenance reports (15 positive, 15 negative) was created manually, covering scenarios like sensor readings, equipment failures, calibration outcomes, and safety alerts. The pre-trained DistilBERT model was fine-tuned on this dataset for 5 epochs using Hugging Face's `Trainer` API.

### Fine-tuned Results (After Fine-tuning)

| Report | Prediction | Confidence | Correct? |
|---|---|---|---|
| The motor is running smoothly with no unusual vibrations detected. | POSITIVE | 99.99% | ✅ |
| Bearing temperature has exceeded safe limits and requires immediate attention. | NEGATIVE | 99.97% | ✅ |
| Routine inspection completed, all systems operating within normal parameters. | POSITIVE | 99.99% | ✅ |
| Critical failure detected in the cooling system, production halted. | NEGATIVE | 99.98% | ✅ |
| The robotic arm calibration was successful and performance has improved significantly. | POSITIVE | 99.99% | ✅ |
| Insulation damage found on multiple cables, urgent replacement needed. | NEGATIVE | 99.99% | ✅ |

**Fine-tuned accuracy on industrial text: 6/6 (100%)**

### A note on the 100% figures

Both the validation accuracy during training and the test accuracy above show 100%. This is expected given the very small dataset size (30 samples, 6 used for validation) and is **not a claim of a production-grade, generalized model** — it's a small-scale demonstration of how domain-specific fine-tuning corrects systematic misclassification. A production version would need a much larger, more diverse labeled dataset to be reliable at scale.

---

## Model Architecture

- **Base model:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- **Task:** Binary sequence classification (Positive / Negative)
- **Fine-tuning method:** Full fine-tuning via Hugging Face `Trainer` API
- **Epochs:** 5
- **Batch size:** 4 (train and eval)
- **Tokenization:** Max length 64, truncation and padding enabled

---

## Project Structure

```
HuggingFace-Sentiment-Analysis/
│
├── main.py                      # Dataset loading + baseline model testing
├── industrial_data.py           # Custom labeled industrial dataset (30 samples)
├── finetune.py                  # Fine-tuning script
├── compare_results.py           # Runs fine-tuned model on test sentences
├── baseline_results.txt         # Saved baseline model predictions
├── finetuned_results.txt        # Saved fine-tuned model predictions
├── finetuned_industrial_model/  # Saved fine-tuned model (gitignored)
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Set up environment**
```bash
python -m venv hf_env
hf_env\Scripts\activate      # Windows
pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn huggingface-hub accelerate
```

2. **Run baseline test**
```bash
python main.py
```

3. **Verify the custom dataset**
```bash
python industrial_data.py
```

4. **Fine-tune the model**
```bash
python finetune.py
```

5. **Compare results (after fine-tuning)**
```bash
python compare_results.py
```

---

## Tech Stack

- Python 3.11
- Hugging Face Transformers
- PyTorch
- Hugging Face Datasets
- Hugging Face Trainer API

---

## Relevance to Industrial Automation

Automated triage of maintenance logs and technician reports is a real use case in industrial operations — flagging urgent issues automatically from free-text reports can speed up response times in large-scale manufacturing environments. This project demonstrates the core NLP workflow behind such a system: identifying a pre-trained model's domain limitations, building a small labeled dataset, and fine-tuning to close that gap — the same process used to adapt general-purpose language models to specialized industrial vocabulary.

---

## Author

Yash Gupta
GitHub: [YashGupta018](https://github.com/YashGupta018)
