# 🎙️ RAGVoiceAssistant: AI Call Center Agent (STT → RAG → TTS)

## Overview
**RAGVoiceAssistant** is an **AI call center agent** for **شركة الشفاء الرقمية للرعاية الصحية** in Saudi Arabia.  

When a customer calls:  
1. The agent captures **the caller’s voice**.  
2. Converts it to text using **Speech-to-Text (STT)**.  
3. Sends the text to a **RAG pipeline** that retrieves answers from **company documents**.  
4. Generates an **intelligent response** using **`gpt-oss-120b` via OpenRouter**.  
5. Converts the response to **spoken voice in a Saudi Arabic accent** using **Text-to-Speech (TTS)**.  
6. Plays the answer **directly to the caller**, fully automated and in real-time.  

This ensures **accurate, domain-specific, and culturally natural responses** for Saudi customers.

---

## 🔄 System Pipeline

Caller Voice
│
▼
Speech-to-Text (STT)
│
▼
Query Embedding
│
▼
Vector Database (FAISS / Chroma)
│
▼
Large Language Model (LLM: gpt-oss-120b via OpenRouter)
│
▼
Text-to-Speech (TTS in Saudi Arabic accent)
│
▼
Spoken Answer to Caller


> Note: The **Vector Database** uses **company documents from شركة الشفاء الرقمية**, making responses **domain-aware and accurate**.

---

## ✨ Key Features
- 🎤 Real-time **voice queries from callers**  
- 📚 **RAG-based answers** using real company documents  
- 🤖 **`gpt-oss-120b` via OpenRouter** for smart responses  
- 🔊 **Text-to-Speech in Saudi Arabic accent**  
- 🌍 Supports **Arabic (primary)** and English  
- 🧪 Includes notebooks and demos for testing  

---

## 🛠️ Tech Stack
- **Speech-to-Text:** Whisper / Hugging Face  
- **Embeddings:** Sentence Transformers  
- **Vector Store:** FAISS / Chroma  
- **LLM:** `gpt-oss-120b` via OpenRouter  
- **Text-to-Speech:** facebook / tts (Saudi accent)  
- **Language:** Python  

---

## 📁 Repository Structure
RAGVoiceAssistant/
├── Data/ → Company documents & audio samples
├── Dummy/ → Experiments & prototype notebooks
├── NoteBooks/ → STT, RAG, TTS pipelines
├── Samples/ → Audio examples
├── Video/ → Demo recordings
├── .gitignore
└── README.md


---

## 🔐 Security Best Practices
- API keys are **never hard-coded**  
- Secrets loaded via **environment variables**  
- `.env` files ignored via `.gitignore`  

---

## 🎯 Use Cases
- Automated **call center agent** for healthcare  
- Domain-specific customer support  
- Arabic AI assistants **with Saudi accent**  
- Accessibility & knowledge assistance  

---

## 🚀 Future Improvements
- Real-time **telephony integration** (SIP / Twilio / WebRTC)  
- Convert notebooks into **production-ready Python modules**  
- Add **monitoring dashboard** for calls  
- Stream audio with **low latency**  
- Dockerize and deploy to cloud platforms  

---

## 👨‍💻 Author
**Amir Tarek**  
Machine Learning Engineer  
linkedin : https://www.linkedin.com/in/amir-tarek1/
