# 🧠 Evaluating Traditional ML Models vs Transformer Architectures for Hate Speech Detection

## 📌 Overview

This project presents a comparative analysis between **traditional machine learning models** — *Logistic Regression (LR)* and *Multinomial Naive Bayes (MNB)* — and a **Transformer-based architecture (DistilBERT)** for multi-stage **hate speech detection** using the OLID dataset.
The goal is to evaluate how **deep contextual embeddings** improve classification accuracy and contextual understanding compared to conventional feature-based approaches.

---

## 🎯 Problem Statement

Social media platforms like Twitter and Facebook have led to an exponential increase in user-generated content, including hate speech and offensive remarks. Manual moderation is infeasible at scale.

To address this, a **two-stage hate speech detection pipeline** was developed:

1. **Stage 1:** Detect *Offensive* vs *Non-Offensive* tweets.
2. **Stage 2A:** For offensive tweets, classify as *Targeted* (TIN) or *Untargeted* (UNT).
3. **Stage 2B:** For targeted tweets, identify target type — *Individual (IND)*, *Group (GRP)*, or *Others (OTH)*.

---

## 🧾 Dataset

**Dataset Used:** [OLID – Offensive Language Identification Dataset (Kaggle)](https://www.kaggle.com/datasets/olid)

| Stage                | Labels                             | Count                  | Observation                              |
| -------------------- | ---------------------------------- | ---------------------- | ---------------------------------------- |
| Stage 1 (Subtask A)  | NOT: 8,840 / OFF: 4,400            | Moderately imbalanced  | Twice as many non-offensive samples      |
| Stage 2A (Subtask B) | TIN: 3,876 / UNT: 524              | Highly imbalanced      | Majority offensive tweets are targeted   |
| Stage 2B (Subtask C) | IND: 2,407 / GRP: 1,074 / OTH: 395 | Complex class overlaps | Most offensive tweets target individuals |

Preprocessing involved cleaning tweets (removing URLs, mentions, hashtags, special symbols), converting to lowercase, and applying **minority class upsampling** to balance training.

---

## ⚙️ Methodology

### 🔹 Traditional Machine Learning Models

* **Logistic Regression (LR):** TF-IDF feature-based linear model.
* **Multinomial Naive Bayes (MNB):** Word-frequency-based probabilistic classifier.

**Workflow:**
`Text → TF-IDF Vectorization → Classifier (LR/MNB) → Output`

### 🔹 Transformer Model – DistilBERT

A compact version of BERT retaining ~97% of performance while being faster and lighter. Fine-tuned individually for each stage.

**Workflow:**
`Text → Tokenizer → DistilBERT Encoder → Classification Layer → Output`

### 🧮 Evaluation Metrics

* Accuracy
* Macro & Weighted Precision, Recall, F1-score

---

## 📊 Results and Analysis

| **Stages**   | **Models**     | **Macro Averaging** |            |          | **Weighted Averaging** |            |          | **Accuracy** |
| ------------ | -------------- | ------------------- | ---------- | -------- | ---------------------- | ---------- | -------- | ------------ |
|              |                | **Precision**       | **Recall** | **F1**   | **Precision**          | **Recall** | **F1**   |              |
| **Stage-1**  | LG             | 0.66                | 0.67       | 0.66     | 0.70                   | 0.69       | 0.69     | 0.69         |
|              | MNB            | 0.66                | 0.66       | 0.66     | 0.70                   | 0.70       | 0.70     | 0.70         |
|              | **DistilBERT** | **0.74**            | **0.75**   | **0.75** | **0.78**               | **0.77**   | **0.77** | **0.77**     |
| **Stage-2A** | LG             | 0.56                | 0.56       | 0.56     | 0.82                   | 0.82       | 0.82     | 0.82         |
|              | MNB            | **0.60**            | 0.53       | 0.53     | 0.82                   | **0.87**   | **0.83** | **0.87**     |
|              | DistilBERT     | 0.55                | 0.53       | 0.53     | 0.80                   | 0.84       | 0.82     | 0.84         |
| **Stage-2B** | LG             | 0.52                | 0.52       | 0.52     | 0.67                   | 0.66       | 0.66     | 0.66         |
|              | MNB            | 0.45                | 0.47       | 0.45     | 0.62                   | 0.66       | 0.64     | 0.66         |
|              | **DistilBERT** | **0.54**            | **0.55**   | **0.54** | **0.69**               | **0.69**   | **0.69** | **0.69**     |

---

## 🔍 Comparative Insights

* **DistilBERT** consistently achieved superior contextual understanding and overall accuracy.
* **Stage 1:** Highest Macro F1 = 0.75, Accuracy = 0.77.
* **Stage 2A:** MNB slightly led due to clear lexical patterns (Accuracy = 0.87).
* **Stage 2B:** DistilBERT again led (Macro F1 = 0.54, Accuracy = 0.69).
* **Conclusion:** Transformers excel at context-based hate speech detection, while traditional models remain efficient for simpler, interpretable tasks.

---

## 🧩 Repository Structure

```
├── LG AND MNB Implementation.ipynb   # Logistic Regression & Naive Bayes models
├── DistilBERT Implementation.ipynb    # Transformer fine-tuning & evaluation
├── GRP-13-MTECH.pdf                   # Detailed project report
└── README.md                          # Documentation file
```

---

## 🧠 Conclusion

DistilBERT achieved the **best trade-off between accuracy and contextual comprehension**, capturing subtle and implicit hate speech patterns.
Logistic Regression offered **interpretability and consistency**, while Naive Bayes proved **efficient for lightweight text classification**.
Together, they form a balanced approach for scalable and explainable **hate speech detection systems**.

---

## 👥 Authors

**Group 13 – CS683 Project (IIIT Guwahati)**

* Mayank Singh (2402055)
* Nishant Kashyap (2402063)
* Shreya Ghosh (2402029)
* Hrishiraj Sawan (2402012)
* Sunidhi Choudhary (2402035)

**Project Guide:** *Dr. Kuntal Dey*
Department of Computer Science & Engineering
Indian Institute of Information Technology, Guwahati

---

## ⚙️ Tech Stack

* Python (scikit-learn, pandas, numpy)
* Hugging Face Transformers (DistilBERT)
* Jupyter Notebook
* Kaggle OLID Dataset

---

## 📚 Citation

> Group 13 (2025). *Evaluating Traditional ML Models against Transformer Architectures for Hate Speech Severity and Target Detection.* IIIT Guwahati, India.
