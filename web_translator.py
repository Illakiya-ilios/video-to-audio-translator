from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

RATE = 16000
CHUNK = int(RATE / 10)

speech_client = speech.SpeechClient()
translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()

audio_queue = queue.Queue()

# Translation state
translation_state = {
    'active': False,
    'source_lang': 'fr',
    'target_lang': 'en',
    'source_lang_code': 'fr-FR',
    'source_lang_name': 'French',
    'target_lang_name': 'English'
}

stream = None


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    if translation_state['active']:
        audio_queue.put(indata.copy())


def audio_generator():
    while translation_state['active']:
        try:
            chunk = audio_queue.get(timeout=1)
            yield speech.StreamingRecognizeRequest(
                audio_content=chunk.tobytes()
            )
        except queue.Empty:
            continue


def speak_text(text, lang_code):
    """Generate and play audio"""
    synthesis_input = texttospeech.SynthesisInput(text=text)

    if lang_code == "fr":
        voice = texttospeech.VoiceSelectionParams(
            language_code="fr-FR",
            name="fr-FR-Neural2-B"
        )
    else:
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-D"
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


def run_streaming():
    """Background streaming translation"""
    global stream
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=translation_state['source_lang_code'],
        model="default",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )

    last_transcript = ""

    def translate_and_speak(text):
        try:
            socketio.emit('translation_status', {'status': 'translating'})
            
            translated = translate_client.translate(
                text,
                source_language=translation_state['source_lang'],
                target_language=translation_state['target_lang']
            )

            translated_text = translated["translatedText"]
            
            socketio.emit('translation_result', {
                'source': text,
                'target': translated_text,
                'source_lang': translation_state['source_lang_name'],
                'target_lang': translation_state['target_lang_name']
            })

            speak_text(translated_text, translation_state['target_lang'])
            
        except Exception as e:
            socketio.emit('error', {'message': str(e)})

    stream = sd.InputStream(
        samplerate=RATE,
        blocksize=CHUNK,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    )
    
    stream.start()

    try:
        requests = audio_generator()
        responses = speech_client.streaming_recognize(
            streaming_config,
            requests,
        )

        for response in responses:
            if not translation_state['active']:
                break
                
            if not response.results:
                continue
                
            for result in response.results:
                if not result.alternatives:
                    continue
                    
                transcript = result.alternatives[0].transcript
                
                if not transcript.strip():
                    continue

                if result.is_final:
                    socketio.emit('transcript', {
                        'text': transcript,
                        'is_final': True
                    })
                    
                    if transcript.strip() and transcript != last_transcript:
                        last_transcript = transcript
                        
                        thread = threading.Thread(
                            target=translate_and_speak, 
                            args=(transcript,),
                            daemon=True
                        )
                        thread.start()
                    
                else:
                    socketio.emit('transcript', {
                        'text': transcript,
                        'is_final': False
                    })
                    
    except Exception as e:
        socketio.emit('error', {'message': str(e)})
    finally:
        if stream:
            stream.stop()
            stream.close()


@app.route('/')
def index():
    return render_template('translator.html')


@socketio.on('start_translation')
def handle_start(data):
    global translation_state
    
    direction = data.get('direction', 'fr-en')
    
    if direction == 'fr-en':
        translation_state.update({
            'source_lang': 'fr',
            'target_lang': 'en',
            'source_lang_code': 'fr-FR',
            'source_lang_name': 'French',
            'target_lang_name': 'English'
        })
    else:
        translation_state.update({
            'source_lang': 'en',
            'target_lang': 'fr',
            'source_lang_code': 'en-US',
            'source_lang_name': 'English',
            'target_lang_name': 'French'
        })
    
    translation_state['active'] = True
    
    # Clear audio queue
    while not audio_queue.empty():
        audio_queue.get()
    
    emit('status', {
        'active': True,
        'direction': f"{translation_state['source_lang_name']} → {translation_state['target_lang_name']}"
    })
    
    # Start streaming in background thread
    thread = threading.Thread(target=run_streaming, daemon=True)
    thread.start()


@socketio.on('stop_translation')
def handle_stop():
    global translation_state, stream
    translation_state['active'] = False
    
    if stream:
        stream.stop()
        stream.close()
        stream = None
    
    emit('status', {'active': False})


@socketio.on('change_direction')
def handle_change_direction(data):
    # Stop current translation
    handle_stop()
    
    # Wait a moment
    time.sleep(0.5)
    
    # Start with new direction
    handle_start(data)


if __name__ == '__main__':
    print("=" * 60)
    print("  LIVE MEETING TRANSLATOR - WEB UI")
    print("=" * 60)
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\n💡 You can switch translation direction on the fly!")
    print("Press Ctrl+C to stop\n")
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
