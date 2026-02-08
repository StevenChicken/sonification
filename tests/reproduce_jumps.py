import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jazz_harmonizer import build_jazz_chord_track, NOTE_NAMES

def test_voice_leading():
    # Create a melody that jumps: C4 -> C5 -> C4
    # Chords should ideally stay somewhat stable or move smoothly, 
    # but current implementation "shadows" the melody.
    
    notes = [60, 72, 60] # C4, C5, C4
    durations = [4.0, 4.0, 4.0] # 1 bar each
    root = "C"
    mode = "IONIAN"
    
    print("--- Generating Chords for Jumping Melody ---")
    track = build_jazz_chord_track(notes, durations, root, mode)
    
    prev_voicing = None
    for beat, voicing, name, dur in track:
        print(f"Chord: {name}, Voicing: {voicing}")
        if prev_voicing:
            diffs = [n2 - n1 for n1, n2 in zip(prev_voicing, voicing)]
            abs_diff_sum = sum(abs(d) for d in diffs)
            print(f"  -> Movement: {diffs} (Total Abs: {abs_diff_sum})")
        prev_voicing = voicing

if __name__ == "__main__":
    test_voice_leading()
