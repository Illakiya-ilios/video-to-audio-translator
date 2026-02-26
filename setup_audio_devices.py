"""
Audio Device Setup Helper
Run this first to find your correct device IDs
"""

import sounddevice as sd

print("=" * 70)
print("  AUDIO DEVICE SETUP")
print("=" * 70)

print("\n📋 Available Audio Devices:\n")

devices = sd.query_devices()

for i, device in enumerate(devices):
    device_type = []
    if device['max_input_channels'] > 0:
        device_type.append('INPUT')
    if device['max_output_channels'] > 0:
        device_type.append('OUTPUT')
    
    type_str = ' & '.join(device_type)
    
    print(f"[{i}] {device['name']}")
    print(f"    Type: {type_str}")
    print(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
    print(f"    Sample Rate: {device['default_samplerate']} Hz")
    print()

print("=" * 70)
print("  SETUP INSTRUCTIONS")
print("=" * 70)

print("\n🎤 For your architecture:")
print("\n1. INPUT DEVICE (Capture Meet audio):")
print("   - Should be: CABLE Output (VB-Audio Virtual Cable)")
print("   - Look for device with 'CABLE Output' or 'VB-Audio' in name")
print("   - Must have INPUT channels > 0")

print("\n2. OUTPUT DEVICE (Send translated audio to Meet):")
print("   - Should be: CABLE Input (VB-Audio Virtual Cable)")
print("   - Look for device with 'CABLE Input' or 'VB-Audio' in name")
print("   - Must have OUTPUT channels > 0")

print("\n📝 Once you identify the devices:")
print("   1. Note the device numbers [X]")
print("   2. Open web_translator.py")
print("   3. Update these lines:")
print("      INPUT_DEVICE = X   # Your CABLE Output device number")
print("      OUTPUT_DEVICE = Y  # Your CABLE Input device number")

print("\n💡 Example:")
print("   If CABLE Output is device [5] and CABLE Input is device [6]:")
print("   INPUT_DEVICE = 5")
print("   OUTPUT_DEVICE = 6")

print("\n" + "=" * 70)
