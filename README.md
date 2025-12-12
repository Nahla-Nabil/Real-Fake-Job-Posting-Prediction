# 🛡️ JobGuard – Fake Job Posting Detector

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-NLP-orange.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success.svg)](https://huggingface.co/spaces/NahlaNabil/job-fraud-classifier)

> **JobGuard** is an AI-powered web application that detects **fraudulent job postings** using Natural Language Processing (NLP) and a fine-tuned Transformer model.

---

## 🌟 Overview

Online job scams have become increasingly common, exposing job seekers to financial loss and identity theft.  
**JobGuard** helps users evaluate job descriptions by analyzing linguistic patterns commonly found in fraudulent postings.

The system provides:
- A fraud risk classification (High / Low)
- Confidence score
- Explainable AI insights (*Why this result?*)
- Safety recommendations for users

---

## 🚀 Live Demo

👉 **Try JobGuard here:**  
🔗 https://huggingface.co/spaces/NahlaNabil/job-fraud-classifier

---

## ✨ Features

- **Fake vs Real Classification** using a Transformer-based model  
- **Confidence Score** for prediction reliability  
- **Suspicious Keyword Detection** (fees, urgency, easy money, etc.)  
- **Explainable AI Panel** explaining why a job is risky or safe  
- **Professional UI** built with Streamlit  
- **Ready-to-use Web App** (no installation needed for users)

---

## 🧠 How It Works

1. The user pastes a job description into the application  
2. The text is tokenized and processed by a fine-tuned **DistilBERT** model  
3. The model predicts whether the job is **Real or Fraudulent**  
4. Additional rule-based signals highlight risk factors  
5. The system explains the decision and provides safety recommendations  

---

## 🏗️ System Architecture
Job Description Text
↓
Tokenization & Preprocessing
↓
DistilBERT Transformer Model
↓
Classification (Real / Fake)
↓
Confidence Score + Explainable Signals


---

## 🛠️ Tech Stack

- **Language:** Python  
- **Model:** DistilBERT (Transformer-based NLP model)  
- **Frameworks:** Hugging Face Transformers, PyTorch  
- **Frontend:** Streamlit  
- **Deployment:** Hugging Face Spaces  

---

## 📁 Project Structure

job-fraud-classifier/
│
├── app.py # Streamlit application
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
├── assets/
│ └── jobguard.png # Project image


> ⚠️ Model weights are hosted on Hugging Face Hub and loaded dynamically.

---

## ⚠️ Responsible AI Notice

JobGuard provides a **risk estimation** based on learned patterns from historical data.  
It should **not** be used as a final decision-making tool.  
Users are encouraged to verify employers through official and trusted channels.

---

## 🔮 Future Improvements

- Multi-language support (Arabic, Spanish, etc.)
- Advanced explainability (SHAP / LIME)
- API for job platforms and recruiters
- Browser extension for real-time job scanning
- Continuous retraining with new scam patterns

---

## 👩‍💻 Author

**Team Leader: Nahla Nabil**

**Group 1**

**Samsung Innovation Campus 2025**

- GitHub: https://github.com/NahlaNabil  
- Hugging Face: https://huggingface.co/NahlaNabil  

---

## 🙏 Acknowledgments

- Dataset: *Kaggle – Real or Fake Job Posting Prediction*  
- Libraries: Hugging Face Transformers, PyTorch, Streamlit  
- Inspiration: NLP research on online fraud detection  

---

<div align="center">

🛡️ **JobGuard – Protecting Job Seekers with AI**  

</div>










