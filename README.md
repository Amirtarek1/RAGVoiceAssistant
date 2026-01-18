# 🎙️ RAG Voice Assistant (STT → RAG → TTS)

## Overview
This project is an **end-to-end voice-based AI assistant** that allows users to ask questions using **speech**, retrieves relevant information from **real-world documents**, and responds back with **spoken answers**.

The system uses **company data from شركة الشفاء الرقمية للرعاية الصحية** as the knowledge base for the **RAG pipeline**, making it capable of answering **domain-specific questions about the company**.  

The project combines **Speech-to-Text (STT)**, **Vector Search**, **Large Language Models (LLMs)**, specifically **`gpt-oss-120b` from OpenRouter**, and **Text-to-Speech (TTS)** into a single pipeline.

---

## 🔄 System Pipeline
User Voice  
→ Speech-to-Text (STT)  
→ Query Embedding  
→ Vector Database (FAISS / Chroma)  
→ Large Language Model (LLM: `gpt-oss-120b` via OpenRouter)  
→ Text-to-Speech (TTS)  
→ Audio Response  

---

## ✨ Key Features
- 🎤 Voice-based query input (STT)
- 📚 Document-aware responses using **RAG** with real healthcare data from شركة الشفاء الرقمية للرعاية الصحية
- 🤖 LLM-powered answer generation using **`gpt-oss-120b` from OpenRouter**
- 🔊 Spoken AI responses (TTS)
- 🌍 Supports Arabic & English (model-dependent)
- 🧪 Includes experiments and demo notebooks

---

## 🛠️ Tech Stack
- **Speech-to-Text:** Whisper / Hugging Face
- **Embeddings:** Sentence Transformers
- **Vector Store:** FAISS / Chroma
- **LLM:** `gpt-oss-120b` via OpenRouter
- **Text-to-Speech:** facebook / tts
- **Language:** Python

---

## 📁 Repository Structure
RAGVoiceAssistant/
- Data/        → Documents & audio samples (including healthcare data from شركة الشفاء الرقمية للرعاية الصحية)  
- Dummy/       → Experiments & prototypes  
- NoteBooks/   → STT, RAG, TTS pipelines  
- Samples/     → Audio examples  
- Video/       → Demo recordings  

---

## 🔐 Security Best Practices
- API keys are **never hard-coded**
- Secrets are loaded via **environment variables**
- `.env` files are excluded using `.gitignore`

---

## 🎯 Use Cases
- Voice-based document question answering
- Domain-specific healthcare assistants
- Arabic AI assistants
- Accessibility tools
- Knowledge assistants
- Educational and enterprise AI systems

---

## 🚀 Future Work
- Convert notebooks into production-ready Python modules
- Add FastAPI backend and Gradio UI
- Support real-time streaming audio
- Dockerize and deploy to cloud platforms

---

## 👨‍💻 Author
**Amir Tarek**  
Machine Learning Engineer  
linked in : https://www.linkedin.com/in/amir-tarek1/
