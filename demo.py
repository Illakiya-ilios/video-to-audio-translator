import vertexai
from moviepy import VideoFileClip, AudioFileClip
from google.cloud import texttospeech_v1 as texttospeech
from google.cloud import speech_v1 as speech
from google.cloud import translate_v2 as translate
import os
import html

# ========================
# CONFIG
# ========================

PROJECT_ID = "iliosdigital-ai-poc"
LOCATION = "us-central1"

INPUT_VIDEO = "videoplayback.mp4"
TEMP_AUDIO = "tamil_audio.wav"

OUTPUT_AUDIO = "english_audio.mp3"

# Male voice options:
# "en-US-Neural2-D" - Standard male
# "en-US-Neural2-J" - Deep male
# "en-US-Studio-M" - Studio quality male
VOICE_NAME = "en-US-Neural2-D"

# ========================
# INIT Vertex AI
# ========================

vertexai.init(project=PROJECT_ID, location=LOCATION)


# ========================
# STEP 1: Extract audio
# ========================

def extract_audio(video_path, audio_path):

    print("Extracting Tamil audio...")

    with VideoFileClip(video_path) as video:

        video.audio.write_audiofile(
            audio_path,
            fps=16000,
            nbytes=2,
            ffmpeg_params=["-ac", "1"]
        )

    print("Audio extracted:", audio_path)


# ========================
# STEP 2: Speech to Text (Tamil)
# ========================

def speech_to_text(audio_file):
    print("Transcribing Tamil audio...")
    
    from google.cloud import storage
    
    # Upload to GCS for long audio files
    storage_client = storage.Client()
    bucket_name = "ilios-speech-audio"
    blob_name = "temp_tamil_audio.wav"
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(audio_file)
    
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded to {gcs_uri}")
    
    # Use standard speech client
    speech_client = speech.SpeechClient()
    
    audio = speech.RecognitionAudio(uri=gcs_uri)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="ta-IN",
        enable_automatic_punctuation=True,
        model="latest_long",
        use_enhanced=True,
        audio_channel_count=1,
        enable_word_time_offsets=True,
        enable_word_confidence=True,
        max_alternatives=1,
    )
    
    print("Processing audio from GCS (this will take several minutes)...")
    
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    
    print("Waiting for transcription... (be patient, 12-min audio takes time)")
    response = operation.result(timeout=3600)
    
    full_text = ""
    word_count = 0
    for result in response.results:
        transcript = result.alternatives[0].transcript
        full_text += transcript + " "
        word_count += len(transcript.split())
    
    print(f"✅ Transcribed {word_count} words ({len(full_text)} characters)")
    return full_text.strip()


# ========================
# STEP 3: Translate Tamil to English
# ========================

def translate_text(tamil_text):
    print("Translating to English...")
    
    translate_client = translate.Client()
    result = translate_client.translate(
        tamil_text,
        source_language="ta",
        target_language="en"
    )
    
    english_text = html.unescape(result["translatedText"])
    print(f"Translated {len(english_text)} characters")
    return english_text


# ========================
# STEP 4: Text to Speech (English)
# ========================

def text_to_speech(text, output_audio):
    print("Generating English audio...")
    
    tts_client = texttospeech.TextToSpeechClient()
    
    # Split into chunks if text is too long
    max_chars = 4000
    chunks = []
    current = ""
    
    sentences = text.split(". ")
    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
    
    if current:
        chunks.append(current.strip())
    
    print(f"Processing {len(chunks)} audio chunks...")
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=VOICE_NAME
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.95,
        pitch=-2.0
    )
    
    audio_clips = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}")
        
        synthesis_input = texttospeech.SynthesisInput(text=chunk)
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        temp_file = f"temp_chunk_{i}.mp3"
        with open(temp_file, "wb") as out:
            out.write(response.audio_content)
        
        audio_clips.append(AudioFileClip(temp_file))
    
    # Concatenate if multiple chunks
    if len(audio_clips) > 1:
        from moviepy import concatenate_audioclips
        final_audio = concatenate_audioclips(audio_clips)
        final_audio.write_audiofile(output_audio, codec='libmp3lame')
        final_audio.close()
    else:
        audio_clips[0].write_audiofile(output_audio, codec='libmp3lame')
        audio_clips[0].close()
    
    # Cleanup
    for i in range(len(chunks)):
        temp_file = f"temp_chunk_{i}.mp3"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"English audio saved: {output_audio}")


# ========================
# STEP 5: Merge audio with video
# ========================

def merge_audio_video(input_video, new_audio, output_video):

    print("Merging English audio with video...")

    video = VideoFileClip(input_video)
    audio = AudioFileClip(new_audio)

    final = video.with_audio(audio)

    final.write_videofile(
        output_video,
        codec="libx264",
        audio_codec="aac"
    )
    
    video.close()
    audio.close()
    final.close()

    print("Final video saved:", output_video)


# ========================
# PIPELINE
# ========================

def process():
    
    # Extract audio from video
    extract_audio(INPUT_VIDEO, TEMP_AUDIO)
    
    # Get original duration
    original_audio = AudioFileClip(TEMP_AUDIO)
    original_duration = original_audio.duration
    original_audio.close()
    print(f"\nOriginal audio: {original_duration/60:.2f} minutes")
    
    # Transcribe Tamil speech
    tamil_text = speech_to_text(TEMP_AUDIO)
    
    # Translate to English
    english_text = translate_text(tamil_text)
    
    # Generate English speech
    text_to_speech(english_text, OUTPUT_AUDIO)
    
    # Check generated duration
    generated_audio = AudioFileClip(OUTPUT_AUDIO)
    generated_duration = generated_audio.duration
    generated_audio.close()
    print(f"Generated audio: {generated_duration/60:.2f} minutes")
    print(f"Duration difference: {abs(original_duration - generated_duration)/60:.2f} minutes")

    print("\n✅ DONE. English audio ready:", OUTPUT_AUDIO)


# ========================
# RUN
# ========================

if __name__ == "__main__":
    process()
