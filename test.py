import vertexai
from moviepy import VideoFileClip
from moviepy import AudioFileClip, concatenate_audioclips
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from google.cloud import storage
import os
import re
import html
# -------------------------
# CONFIG
# -------------------------
PROJECT_ID = "iliosdigital-ai-poc"  # Update this with your actual project ID
LOCATION = "us-central1"

INPUT_VIDEO = "videoplayback.mp4"
TEMP_AUDIO = "temp_tamil_audio.wav"
OUTPUT_AUDIO = "english_output_audio.mp3"


# -------------------------
# INIT Vertex AI
# -------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)


def upload_to_gcs(bucket_name, source_file, destination_blob):

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(source_file)

    print(f"Uploaded to gs://{bucket_name}/{destination_blob}")

    return f"gs://{bucket_name}/{destination_blob}"

# -------------------------
# STEP 1: Extract Audio
# -------------------------
def extract_audio(video_path, audio_path):

    with VideoFileClip(video_path) as video:

        if video.audio is None:
            raise Exception("❌ This video contains NO audio track")

        video.audio.write_audiofile(
            audio_path,
            fps=16000,        # Required sample rate
            nbytes=2,
            bitrate="64k",
            ffmpeg_params=["-ac", "1"]   # ⭐ THIS makes it mono
        )


# -------------------------
# STEP 2: French Speech → Text
# -------------------------
def speech_to_text(audio_file):

    from google.cloud.speech_v1.services.speech.transports import SpeechRestTransport

    transport = SpeechRestTransport()
    speech_client = speech.SpeechClient(transport=transport)

    bucket_name = "ilios-speech-audio"
    blob_name = "temp_audio.wav"

    gcs_uri = upload_to_gcs(bucket_name, audio_file, blob_name)

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


    print("Processing Tamil audio from GCS...")

    operation = speech_client.long_running_recognize(
        config=config,
        audio=audio
    )

    response = operation.result(timeout=3600)

    full_text = ""
    word_count = 0

    for result in response.results:
        transcript = result.alternatives[0].transcript
        confidence = result.alternatives[0].confidence if hasattr(result.alternatives[0], 'confidence') else 0
        full_text += transcript + " "
        word_count += len(transcript.split())
        
    print(f"Transcribed {word_count} words with average confidence")
    
    return full_text.strip()

# -------------------------
# STEP 3: Translate Tamil → English
# -------------------------
def translate_to_english(tamil_text):

    translate_client = translate.Client()

    result = translate_client.translate(
        tamil_text,
        source_language="ta",
        target_language="en"
    )
    
    # Decode HTML entities like &#39; to '
    translated_text = html.unescape(result["translatedText"])
    
    return translated_text

# -------------------------
# STEP 4: English Text → Speech
# -------------------------
def text_to_speech(long_text, output_file):
    
    # Clean and normalize the text
    long_text = html.unescape(long_text)
    
    # Split text into sentences - handle multiple punctuation marks
    sentences = re.split(r'(?<=[.!?])\s+', long_text)
    
    # Group sentences into chunks of max 4000 characters (safer limit)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is too long, split it further
        if len(sentence) > 4000:
            # Split by commas or other natural breaks
            sub_sentences = re.split(r'(?<=[,;:])\s+', sentence)
            for sub in sub_sentences:
                if len(current_chunk) + len(sub) + 1 <= 4000:
                    current_chunk += sub + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sub + " "
        else:
            if len(current_chunk) + len(sentence) + 1 <= 4000:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Total chunks: {len(chunks)}")
    
    tts_client = texttospeech.TextToSpeechClient()
    
    # Use a male English voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-D"  # Male voice - clear and natural
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.95,  # Slightly slower to better match Tamil pacing
        pitch=-2.0  # Slightly lower pitch for male voice
    )
    
    audio_clips = []
    
    for i, chunk in enumerate(chunks):
        print(f"Generating chunk {i+1}/{len(chunks)}")
        
        try:
            input_text = texttospeech.SynthesisInput(text=chunk)
            
            response = tts_client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config
            )
            
            temp_file = f"temp_chunk_{i}.mp3"
            with open(temp_file, "wb") as out:
                out.write(response.audio_content)
            
            audio_clips.append(AudioFileClip(temp_file))
            
        except Exception as e:
            print(f"Error generating chunk {i+1}: {e}")
            print(f"Chunk length: {len(chunk)} characters")
            print(f"Chunk preview: {chunk[:100]}...")
            raise
    
    # Concatenate all audio clips
    if len(audio_clips) > 1:
        print("Concatenating audio chunks...")
        final_audio = concatenate_audioclips(audio_clips)
        final_audio.write_audiofile(output_file, codec='libmp3lame')
        final_audio.close()
    else:
        audio_clips[0].write_audiofile(output_file, codec='libmp3lame')
        audio_clips[0].close()
    
    # Clean up temp files
    for i in range(len(chunks)):
        temp_file = f"temp_chunk_{i}.mp3"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("✅ FINAL AUDIO GENERATED:", output_file)


# -------------------------
# PIPELINE RUNNER
# -------------------------
def process_video():

    print("Extracting Audio...")
    extract_audio(INPUT_VIDEO, TEMP_AUDIO)
    
    # Get original audio duration
    original_audio = AudioFileClip(TEMP_AUDIO)
    original_duration = original_audio.duration
    original_audio.close()
    print(f"Original audio duration: {original_duration:.2f} seconds ({original_duration/60:.2f} minutes)")

    print("Converting Tamil Speech → Text...")
    tamil_text = speech_to_text(TEMP_AUDIO)
    print(f"Tamil Text ({len(tamil_text)} characters):", tamil_text[:200], "...")

    print("Translating → English...")
    english_text = translate_to_english(tamil_text)
    print(f"English Text ({len(english_text)} characters):", english_text[:200], "...")

    print("Generating English Audio...")
    text_to_speech(english_text, OUTPUT_AUDIO)
    
    # Check generated audio duration
    generated_audio = AudioFileClip(OUTPUT_AUDIO)
    generated_duration = generated_audio.duration
    generated_audio.close()
    print(f"Generated audio duration: {generated_duration:.2f} seconds ({generated_duration/60:.2f} minutes)")
    print(f"Duration difference: {abs(original_duration - generated_duration):.2f} seconds")

    print("✅ Completed! Output saved:", OUTPUT_AUDIO)


# -------------------------
# EXECUTE
# -------------------------
if __name__ == "__main__":
    process_video()
