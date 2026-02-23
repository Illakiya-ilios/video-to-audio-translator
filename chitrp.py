import sounddevice as sd
import numpy as np
import queue
import threading
import time
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks

speech_client = speech.SpeechClient()
translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()

audio_queue = queue.Queue()


# ==============================
# AUDIO INPUT STREAM
# ==============================
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def audio_generator():
    while True:
        chunk = audio_queue.get()
        yield speech.StreamingRecognizeRequest(
            audio_content=chunk.tobytes()
        )


# ==============================
# TTS + PLAY AUDIO
# ==============================
def speak_text(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Chirp3-HD-Charon"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)

    sd.play(audio_data, RATE)
    sd.wait()


# ==============================
# STREAMING PIPELINE
# ==============================
def run_streaming():

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="fr-FR",
        model="latest_long",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    print("🎤 Live translator started (processing every 10 seconds)...")

    transcript_buffer = {"text": ""}
    lock = threading.Lock()

    # 🔥 Timer thread
    def window_processor():
        while True:
            time.sleep(10)

            with lock:
                text = transcript_buffer["text"]
                transcript_buffer["text"] = ""

            if text.strip():
                print("⏱ 10-sec window triggered")
                print("📝 French:", text)

                translated = translate_client.translate(
                    text,
                    source_language="fr",
                    target_language="en"
                )

                english_text = translated["translatedText"]
                print("🌍 English:", english_text)

                speak_text(english_text)

    # Start timer thread
    threading.Thread(target=window_processor, daemon=True).start()

    with sd.InputStream(
        samplerate=RATE,
        blocksize=CHUNK,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):

        requests = audio_generator()

        responses = speech_client.streaming_recognize(
            streaming_config,
            requests,
        )

        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript

                if transcript:
                    with lock:
                        transcript_buffer["text"] = transcript

                    print("🟡 LIVE:", transcript)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    run_streaming()
