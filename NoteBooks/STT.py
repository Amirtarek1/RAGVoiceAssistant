import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel, PeftConfig
import time

stt_pipeline = None

def get_stt_pipeline():
    global stt_pipeline 
    if stt_pipeline is None: 
        print("🔄 Loading SaudiSTT pipeline...")
        
        try:
            # Load the fine-tuned model configuration
            peft_model_id = "Bruno7/ksa-whisper"
            config = PeftConfig.from_pretrained(peft_model_id)
            
            # Load base model
            base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Load the fine-tuned model
            model = PeftModel.from_pretrained(base_model, peft_model_id)
            
            # Load processor (Whisper uses Processor, not Tokenizer)
            processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)
            
            # Create pipeline with the ACTUAL fine-tuned model
            stt_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,  # Use the fine-tuned model, not base model
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            print("✅ SaudiSTT pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading STT pipeline: {e}")
            # Return a simple fallback pipeline
            stt_pipeline = pipeline(
                "automatic-speech-recognition",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    return stt_pipeline

class SaudiSTT: 
    def __init__(self):
        self.pipe = get_stt_pipeline()

    def transcribe_audio(self, file_path):
        try:
            result = self.pipe(
                file_path,
                generate_kwargs={"language": "arabic", "task": "transcribe"}
            )
            return result["text"]
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return f"خطأ في التعرف على الصوت: {str(e)}"