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

# Translation settings (will be set by user)
SOURCE_LANG = None
TARGET_LANG = None
SOURCE_LANG_CODE = None
TARGET_LANG_NAME = None
SOURCE_LANG_NAME = None


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
def speak_text(text, lang_code):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Choose voice based on target language
    if lang_code == "fr":
        voice = texttospeech.VoiceSelectionParams(
            language_code="fr-FR",
            name="fr-FR-Neural2-B"  # French male voice
        )
    else:  # English
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-D"  # English male voice
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
        language_code=SOURCE_LANG_CODE,
        model="default",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )

    print(f"\n🎤 Live translator started ({SOURCE_LANG_NAME} → {TARGET_LANG_NAME})")
    print("🔴 Speak now - translation happens immediately after you stop speaking")
    print("💡 Press Ctrl+C to stop\n")

    last_transcript = ""

    def translate_and_speak(text):
        """Translate in background thread for faster response"""
        try:
            print(f"\n📝 {SOURCE_LANG_NAME}: {text}")
            print("🌍 Translating...", end='', flush=True)
            
            translated = translate_client.translate(
                text,
                source_language=SOURCE_LANG,
                target_language=TARGET_LANG
            )

            translated_text = translated["translatedText"]
            print(f"\r🗣️  {TARGET_LANG_NAME}: {translated_text}")

            speak_text(translated_text, TARGET_LANG)
            print()  # New line after speaking
            
        except Exception as e:
            print(f"\n❌ Translation error: {e}\n")

    with sd.InputStream(
        samplerate=RATE,
        blocksize=CHUNK,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):

        requests = audio_generator()

        try:
            responses = speech_client.streaming_recognize(
                streaming_config,
                requests,
            )

            for response in responses:
                if not response.results:
                    continue
                    
                for result in response.results:
                    if not result.alternatives:
                        continue
                        
                    transcript = result.alternatives[0].transcript
                    
                    if not transcript.strip():
                        continue

                    if result.is_final:
                        # Got final result - translate immediately!
                        print(f"\n✅ {transcript}")
                        
                        if transcript.strip() and transcript != last_transcript:
                            last_transcript = transcript
                            
                            # Translate in background thread for speed
                            thread = threading.Thread(
                                target=translate_and_speak, 
                                args=(transcript,),
                                daemon=True
                            )
                            thread.start()
                        
                    else:
                        # Interim result - show live transcription
                        print(f"🟡 {transcript}                    ", end='\r', flush=True)
                        
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
            print("Possible issues:")
            print("  - Microphone not working")
            print("  - No speech detected")
            print("  - Network connection issue")
            import traceback
            traceback.print_exc()

# ==============================
# MENU SYSTEM
# ==============================
def show_menu():
    global SOURCE_LANG, TARGET_LANG, SOURCE_LANG_CODE, TARGET_LANG_NAME, SOURCE_LANG_NAME
    
    print("=" * 60)
    print("         LIVE MEETING TRANSLATOR")
    print("=" * 60)
    print("\nChoose translation direction:\n")
    print("  1. French → English  (Client speaks French)")
    print("  2. English → French  (You speak English)")
    print("  3. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            SOURCE_LANG = "fr"
            TARGET_LANG = "en"
            SOURCE_LANG_CODE = "fr-FR"
            SOURCE_LANG_NAME = "French"
            TARGET_LANG_NAME = "English"
            return True
        elif choice == "2":
            SOURCE_LANG = "en"
            TARGET_LANG = "fr"
            SOURCE_LANG_CODE = "en-US"
            SOURCE_LANG_NAME = "English"
            TARGET_LANG_NAME = "French"
            return True
        elif choice == "3":
            print("\n👋 Goodbye!")
            return False
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    
    # Show menu and get user choice
    if not show_menu():
        exit(0)
    
    print("\n" + "=" * 60)
    print(f"Mode: {SOURCE_LANG_NAME} → {TARGET_LANG_NAME}")
    print("=" * 60)
    print("\nTesting microphone...")
    
    # Test microphone
    try:
        devices = sd.query_devices()
        print(f"Available audio devices: {len(devices)}")
        default_input = sd.query_devices(kind='input')
        print(f"Default input device: {default_input['name']}")
        
        # Quick audio level test
        print(f"\nTesting audio levels (speak in {SOURCE_LANG_NAME} for 3 seconds)...")
        test_audio = sd.rec(int(3 * RATE), samplerate=RATE, channels=1, dtype='int16')
        sd.wait()
        
        max_level = np.max(np.abs(test_audio))
        print(f"Max audio level detected: {max_level}")
        
        if max_level < 100:
            print("⚠️  WARNING: Audio level very low! Check:")
            print("   - Microphone is plugged in")
            print("   - Microphone is not muted")
            print("   - Correct microphone is selected in Windows settings")
            print("   - Speak louder or closer to microphone")
        else:
            print("✅ Microphone working!")
            
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        print("Please check your microphone connection")
        exit(1)
    
    print("\nStarting live translation...")
    
    try:
        run_streaming()
    except KeyboardInterrupt:
        print("\n\n👋 Translation stopped")
        print("\nRestart the program to change translation direction.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
