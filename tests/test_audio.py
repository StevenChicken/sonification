import sys
import os
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonification_app import synthesize_wav

def test_audio_gen():
    print("Testing audio generation...")
    notes = [60, 64, 67]
    durations = [0.5, 0.5, 1.0]
    tempo = 120
    
    # Mock chord track: (beat, notes, name, duration)
    chord_track = [
        (0.0, [48, 52, 55], "Cmaj", 2.0)
    ]
    
    wav_data = synthesize_wav(notes, durations, tempo, chord_track)
    print(f"Generated WAV size: {len(wav_data)} bytes")
    
    if len(wav_data) > 44: # Header size
        print("Success: WAV generated.")
    else:
        print("Error: WAV too small.")

if __name__ == "__main__":
    test_audio_gen()
