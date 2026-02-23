import sounddevice as sd
import numpy as np
import queue
import threading
import time
import torch

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

RATE = 16000
CHUNK = int(RATE / 10)

audio_queue = queue.Queue()

# ==============================
# LOAD MODELS (LOCAL)
# ==============================

# Whisper STT
whisper_model = WhisperModel("large-v2", compute_type="int8")

# NLLB Translator
translator_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(translator_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name)

# Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)


# ==============================
# AUDIO INPUT STREAM
# ==============================

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


# ==============================
# TTS + PLAY AUDIO
# ==============================

def speak_text(text):
    wav = tts.tts(text)
    sd.play(np.array(wav), samplerate=22050)
    sd.wait()


# ==============================
# TRANSLATE FUNCTION
# ==============================

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = translation_model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


# ==============================
# STREAMING PIPELINE
# ==============================

def run_streaming():

    print("🎤 Live translator started (FREE open-source version)...")

    transcript_buffer = {"text": ""}
    lock = threading.Lock()

    # 🔥 Timer thread (10 sec window)
    def window_processor():
        while True:
            time.sleep(10)

            with lock:
                text = transcript_buffer["text"]
                transcript_buffer["text"] = ""

            if text.strip():
                print("⏱ 10-sec window triggered")
                print("📝 French:", text)

                english_text = translate_text(text)
                print("🌍 English:", english_text)

                speak_text(english_text)

    threading.Thread(target=window_processor, daemon=True).start()

    with sd.InputStream(
        samplerate=RATE,
        blocksize=CHUNK,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):

        audio_buffer = np.zeros((0,), dtype=np.float32)

        while True:
            chunk = audio_queue.get().flatten()
            audio_buffer = np.concatenate((audio_buffer, chunk))

            # Process every ~3 sec for Whisper
            if len(audio_buffer) > RATE * 3:

                segments, _ = whisper_model.transcribe(
                    audio_buffer,
                    language="fr"
                )

                text = " ".join([seg.text for seg in segments])

                if text.strip():
                    with lock:
                        transcript_buffer["text"] = text

                    print("🟡 LIVE:", text)

                audio_buffer = np.zeros((0,), dtype=np.float32)


if __name__ == "__main__":
    run_streaming()
