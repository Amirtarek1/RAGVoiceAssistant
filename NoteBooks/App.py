# app.py

import streamlit as st
import torch
import soundfile as sf
import tempfile
from transformers import VitsModel, AutoTokenizer
from STT import SaudiSTT   # ملف STT
from ArabicRAGGPT import ArabicRAGGPT_class  # ملف RAG


# ===== تحميل TTS =====
@st.cache_resource
def load_tts_model():
    model_id = "facebook/mms-tts-ara"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    return tokenizer, model

tokenizer, tts_model = load_tts_model()


def text_to_speech(text, file_path="answer.wav"):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        wav = tts_model(**inputs).waveform
    wav = wav.squeeze().cpu().numpy().astype("float32")
    rate = tts_model.config.sampling_rate if hasattr(tts_model.config, "sampling_rate") else 22050
    sf.write(file_path, wav, rate, format="WAV", subtype="PCM_16")
    return file_path, rate


# ===== Streamlit App =====
st.title("🎙️ STT → RAG → TTS Demo")

# Step 1: رفع الصوت
uploaded_file = st.file_uploader("ارفع ملف صوت (wav / mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # حفظ الملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        audio_path = tmpfile.name

    st.audio(audio_path, format="audio/wav")

    # Step 2: STT
    stt_engine = SaudiSTT()
    text = stt_engine.transcribe_audio(audio_path)
    st.success(f"📄 النص المستخرج: {text}")

    # Step 3: RAG
    api_key = st.secrets["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"]  # ضيف الـ API Key في secrets.toml
    rag = ArabicRAGGPT_class("C:\Centraly_Project\Data\info.md", api_key)
    answer = rag.ask_question(text)
    st.info(f"🤖 إجابة الموديل: {answer}")

    # Step 4: TTS
    audio_file, rate = text_to_speech(answer)
    st.audio(audio_file, format="audio/wav")
    with open(audio_file, "rb") as f:
        st.download_button("⬇️ تحميل الصوت", f, file_name="answer.wav")
