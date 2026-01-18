# 🎙️ RAG Voice Assistant (STT → RAG → TTS)

## Overview
This project is an **end-to-end voice-based AI assistant** that allows users to ask questions using **speech**, retrieves relevant information from custom documents using **Retrieval-Augmented Generation (RAG)**, and responds back with **spoken answers**.

The system combines **Speech-to-Text (STT)**, **Vector Search**, **Large Language Models (LLMs)**, and **Text-to-Speech (TTS)** into a single pipeline.

---

## 🔄 System Pipeline
User Voice  
→ Speech-to-Text (STT)  
→ Query Embedding  
→ Vector Database (FAISS / Chroma)  
→ Large Language Model (LLM)  
→ Text-to-Speech (TTS)  
→ Audio Response  

---

## ✨ Key Features
- 🎤 Voice-based query input (STT)
- 📚 Document-aware responses using RAG
- 🤖 LLM-powered answer generation
- 🔊 Spoken AI responses (TTS)
- 🌍 Supports Arabic & English (model-dependent)
- 🧪 Includes experiments and demo notebooks

---

## 🛠️ Tech Stack
- **Speech-to-Text:** Whisper / Hugging Face
- **Embeddings:** Sentence Transformers
- **Vector Store:** FAISS / Chroma
- **LLM:** OpenAI / Hugging Face Models
- **Text-to-Speech:** Coqui / gTTS
- **Language:** Python

---

## 📁 Repository Structure
RAGVoiceAssistant/
- Data/        → Documents & audio samples  
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
GitHub: https://github.com/Amirtarek1
