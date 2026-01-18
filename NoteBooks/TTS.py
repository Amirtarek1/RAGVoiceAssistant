#!pip install transformers scipy soundfile

from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
from IPython.display import Audio

model_id = "facebook/mms-tts-ara"

# تحميل الموديل والتوكنيزر
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

# النص للتجربة
text = "السلام عليكم كيف حالك اليوم؟"

# تجهيز الإدخال
inputs = tokenizer(text, return_tensors="pt")

# توليد الموجة الصوتية
with torch.no_grad():
    wav = model(**inputs).waveform  # tensor [batch, time]

# تحويل الموجة لشكل [time] وبنوع float32
wav = wav.squeeze().cpu().numpy().astype("float32")

# تحديد معدل العينة
rate = model.config.sampling_rate if hasattr(model.config, "sampling_rate") else 22050

# حفظ الملف بصيغة PCM_16
sf.write("output.wav", wav, rate, format="WAV", subtype="PCM_16")

# تشغيل الصوت في Colab
Audio("output.wav", rate=rate)

