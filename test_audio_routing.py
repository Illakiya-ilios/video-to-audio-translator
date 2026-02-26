"""
Test Audio Routing for Meet Integration
This script tests if audio can be captured and played through virtual cables
"""

import sounddevice as sd
import numpy as np
import time

# Device configuration (same as web_translator.py)
INPUT_DEVICE = 2   # CABLE Output - captures Meet audio
OUTPUT_DEVICE = 4  # CABLE Input - sends to Meet

RATE = 16000
DURATION = 3  # seconds

print("=" * 70)
print("  AUDIO ROUTING TEST")
print("=" * 70)

# List devices
print("\n📋 Audio Devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    if i == INPUT_DEVICE:
        print(f"  [{i}] {device['name']} ← INPUT (capturing from here)")
    elif i == OUTPUT_DEVICE:
        print(f"  [{i}] {device['name']} ← OUTPUT (playing to here)")
    else:
        print(f"  [{i}] {device['name']}")

print("\n" + "=" * 70)
print("  TEST 1: Record from CABLE Output")
print("=" * 70)
print(f"\n🎤 Recording {DURATION} seconds from device [{INPUT_DEVICE}]...")
print("💡 Play something in Google Meet or any audio through CABLE Output")
print("   (You have 3 seconds to start playing audio)")

try:
    recording = sd.rec(
        int(DURATION * RATE),
        samplerate=RATE,
        channels=1,
        dtype='int16',
        device=INPUT_DEVICE
    )
    sd.wait()
    
    # Check if we captured audio
    max_amplitude = np.max(np.abs(recording))
    avg_amplitude = np.mean(np.abs(recording))
    
    print(f"\n📊 Recording Stats:")
    print(f"   Max amplitude: {max_amplitude}")
    print(f"   Avg amplitude: {avg_amplitude:.2f}")
    
    if max_amplitude > 100:
        print("   ✅ Audio captured successfully!")
    else:
        print("   ⚠️  Very low audio level - check if audio is playing through CABLE Output")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("  TEST 2: Play to CABLE Input")
print("=" * 70)
print(f"\n🔊 Playing test tone to device [{OUTPUT_DEVICE}]...")
print("💡 If Meet is configured correctly, participants should hear this tone")

try:
    # Generate a simple test tone (440 Hz - A note)
    duration = 2
    frequency = 440
    t = np.linspace(0, duration, int(RATE * duration), False)
    tone = np.sin(frequency * 2 * np.pi * t) * 0.3
    tone = (tone * 32767).astype(np.int16)
    
    sd.play(tone, RATE, device=OUTPUT_DEVICE)
    sd.wait()
    
    print("   ✅ Test tone played successfully!")
    print("   💡 If you're in Meet with mic set to CABLE Input, others should have heard it")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("  TEST 3: Echo Test (Record + Playback)")
print("=" * 70)
print(f"\n🔄 Recording from [{INPUT_DEVICE}] and playing to [{OUTPUT_DEVICE}]...")
print("💡 This simulates the translation flow")

try:
    print("   Recording 3 seconds...")
    recording = sd.rec(
        int(3 * RATE),
        samplerate=RATE,
        channels=1,
        dtype='int16',
        device=INPUT_DEVICE
    )
    sd.wait()
    
    if np.max(np.abs(recording)) > 100:
        print("   Playing back what was recorded...")
        sd.play(recording, RATE, device=OUTPUT_DEVICE)
        sd.wait()
        print("   ✅ Echo test complete!")
    else:
        print("   ⚠️  No audio captured to play back")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print("\n✅ If all tests passed, your audio routing is configured correctly!")
print("\nNext steps:")
print("  1. Start web_translator.py")
print("  2. Open http://localhost:5000 in browser")
print("  3. Configure Google Meet:")
print("     - Microphone: CABLE Input")
print("     - Speakers: CABLE Output")
print("  4. Start translation and test!")
print("\n" + "=" * 70)
