"""
Jazz Harmonization Engine
=========================
Modal quantization (all 7 modes for any root) and Viterbi-based chord
harmonizer with circle-of-fifths weighting, intruder V7 injection,
and resolution bonuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

NOTE_LOOKUP: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

MODAL_DEFINITIONS: dict[str, dict] = {
    "IONIAN":     {"offset": 0,   "tonic_degree": 1},
    "DORIAN":     {"offset": -2,  "tonic_degree": 2},
    "PHRYGIAN":   {"offset": -4,  "tonic_degree": 3},
    "LYDIAN":     {"offset": -5,  "tonic_degree": 4},
    "MIXOLYDIAN": {"offset": -7,  "tonic_degree": 5},
    "AEOLIAN":    {"offset": -9,  "tonic_degree": 6},
    "LOCRIAN":    {"offset": -11, "tonic_degree": 7},
}

CHORD_QUALITIES: dict[str, list[int]] = {
    "maj7":  [0, 4, 7, 11],
    "m7":    [0, 3, 7, 10],
    "dom7":  [0, 4, 7, 10],
    "m7b5":  [0, 3, 6, 10],
}

DEGREE_QUALITY = ["maj7", "m7", "m7", "maj7", "dom7", "m7", "m7b5"]

# Viterbi scoring
INTERVAL_WEIGHTS: dict[int, int] = {
    5: 100, -7: 100,   # up a 4th / down a 5th (circle of fifths)
    -2: 60, 2: 40,     # step motion
    -3: 30, -4: 30,    # minor/major 3rd down
    0: 5,              # same chord
}
INTERVAL_ELSE = -10
BONUS_TARGET_RESOLUTION = 50
PENALTY_UNRESOLVED_DOM = -20
PENALTY_NON_DIATONIC = -15

MIDI_MIN = 55  # G3
MIDI_MAX = 88  # E6


# ──────────────────────────────────────────────
#  Chord dataclass
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Chord:
    root: int           # pitch class 0-11
    quality: str        # "maj7", "m7", "dom7", "m7b5"
    notes: frozenset    # pitch classes in the chord
    name: str           # display name e.g. "Cm7"
    chord_type: str     # "DIATONIC" or "INTRUDER"


# ──────────────────────────────────────────────
#  Bar data for Viterbi
# ──────────────────────────────────────────────

class BarData(NamedTuple):
    beat1_pc: int          # pitch class on beat 1
    beat3_pc: int          # pitch class on beat 3
    all_pcs: frozenset     # all pitch classes in the bar
    start_beat: float      # beat time of bar start
    duration: float        # total bar duration in beats


# ──────────────────────────────────────────────
#  Modal quantizer
# ──────────────────────────────────────────────

def build_modal_scale(root_pc: int, mode_name: str) -> list[int]:
    """Return 7 pitch classes for the given root and mode."""
    mode = MODAL_DEFINITIONS[mode_name]
    parent_root = (root_pc + mode["offset"]) % 12
    return [(parent_root + interval) % 12 for interval in MAJOR_SCALE]


def quantize_to_mode(
    raw_notes: list[float],
    root_name: str,
    mode_name: str,
    midi_min: int = MIDI_MIN,
    midi_max: int = MIDI_MAX,
) -> list[int]:
    """Snap raw MIDI values to the nearest note in the modal scale."""
    root_pc = NOTE_LOOKUP[root_name]
    scale_pcs = build_modal_scale(root_pc, mode_name)
    legal = [n for n in range(midi_min, midi_max + 1) if (n % 12) in scale_pcs]
    if not legal:
        legal = list(range(midi_min, midi_max + 1))
    return [min(legal, key=lambda n: abs(n - raw)) for raw in raw_notes]


# ──────────────────────────────────────────────
#  Chord pool generator
# ──────────────────────────────────────────────

_QUALITY_SUFFIX = {
    "maj7": "maj7",
    "m7": "m7",
    "dom7": "7",
    "m7b5": "m7b5",
}


def get_chord_pool(root_name: str, mode_name: str) -> tuple[list[Chord], Chord]:
    """Build diatonic 7th chords + intruder V7.  Returns (pool, tonic_chord)."""
    root_pc = NOTE_LOOKUP[root_name]
    mode = MODAL_DEFINITIONS[mode_name]
    parent_root = (root_pc + mode["offset"]) % 12

    pool: list[Chord] = []
    for i in range(7):
        cr = (parent_root + MAJOR_SCALE[i]) % 12
        quality = DEGREE_QUALITY[i]
        intervals = CHORD_QUALITIES[quality]
        notes = frozenset((cr + iv) % 12 for iv in intervals)
        name = NOTE_NAMES[cr] + _QUALITY_SUFFIX[quality]
        pool.append(Chord(root=cr, quality=quality, notes=notes,
                          name=name, chord_type="DIATONIC"))

    # Intruder V7 of the user's root
    intruder_root = (root_pc + 7) % 12
    already_in_pool = any(c.root == intruder_root and c.quality == "dom7"
                         for c in pool)
    if not already_in_pool:
        intervals = CHORD_QUALITIES["dom7"]
        notes = frozenset((intruder_root + iv) % 12 for iv in intervals)
        name = NOTE_NAMES[intruder_root] + "7"
        pool.append(Chord(root=intruder_root, quality="dom7", notes=notes,
                          name=name, chord_type="INTRUDER"))

    tonic_chord = pool[mode["tonic_degree"] - 1]
    return pool, tonic_chord


# ──────────────────────────────────────────────
#  Bar extraction
# ──────────────────────────────────────────────

def _beat_times(durations: list[float]) -> list[float]:
    times: list[float] = []
    t = 0.0
    for d in durations:
        times.append(t)
        t += d
    return times


def extract_bars(notes: list[int], durations: list[float]) -> list[BarData]:
    """Divide melody into 4-beat bars and extract harmonic info."""
    beats = _beat_times(durations)
    if not notes:
        return []

    total_beats = beats[-1] + durations[-1]
    bars: list[BarData] = []
    bar_start = 0.0

    while bar_start < total_beats - 0.01:
        bar_end = bar_start + 4.0

        # Collect notes in this bar
        bar_notes: list[int] = []
        beat1_note: int | None = None
        beat3_note: int | None = None
        last_sounding: int | None = None

        for idx in range(len(notes)):
            note_start = beats[idx]
            note_end = note_start + durations[idx]

            # Track last note that started before or at bar_start
            if note_start <= bar_start + 0.01:
                last_sounding = notes[idx]

            # Notes overlapping this bar
            if note_end > bar_start + 0.01 and note_start < bar_end - 0.01:
                bar_notes.append(notes[idx])

            # Beat 1 (bar_start): note sounding at this time
            if note_start <= bar_start + 0.01 and note_end > bar_start + 0.01:
                beat1_note = notes[idx]

            # Beat 3 (bar_start + 2.0): note sounding at this time
            beat3_time = bar_start + 2.0
            if note_start <= beat3_time + 0.01 and note_end > beat3_time + 0.01:
                beat3_note = notes[idx]

        # Fallbacks
        if beat1_note is None:
            beat1_note = last_sounding if last_sounding is not None else (
                bar_notes[0] if bar_notes else notes[0]
            )
        if beat3_note is None:
            beat3_note = beat1_note

        all_pcs = frozenset(n % 12 for n in bar_notes) if bar_notes else frozenset({beat1_note % 12})
        bar_dur = min(bar_end, total_beats) - bar_start

        bars.append(BarData(
            beat1_pc=beat1_note % 12,
            beat3_pc=beat3_note % 12,
            all_pcs=all_pcs,
            start_beat=bar_start,
            duration=bar_dur,
        ))
        bar_start = bar_end

    return bars


# ──────────────────────────────────────────────
#  Candidate filtering
# ──────────────────────────────────────────────

def get_candidates_for_bar(
    beat1_pc: int,
    beat3_pc: int,
    chord_pool: list[Chord],
    bar_melody_pcs: frozenset,
) -> list[Chord]:
    """Return chords from pool that fit the bar's strong-beat pitches."""
    candidates: list[Chord] = []
    for chord in chord_pool:
        # Strong beats must be chord tones
        if beat1_pc not in chord.notes or beat3_pc not in chord.notes:
            continue

        # Intruder clash check: if the minor 3rd above the intruder root
        # (which would be a b7 of the modal tonic) is in the melody,
        # it clashes with the intruder's major 3rd.
        if chord.chord_type == "INTRUDER":
            avoid_pc = (chord.root + 3) % 12
            if avoid_pc in bar_melody_pcs:
                continue

        candidates.append(chord)

    # If nothing matches, fall back to all diatonic chords
    if not candidates:
        candidates = [c for c in chord_pool if c.chord_type == "DIATONIC"]
    # Ultimate fallback
    if not candidates:
        candidates = list(chord_pool)

    return candidates


# ──────────────────────────────────────────────
#  Viterbi scoring
# ──────────────────────────────────────────────

def calculate_transition_score(prev: Chord, curr: Chord, tonic_chord: Chord) -> int:
    """Score a chord-to-chord transition using jazz logic."""
    raw_interval = (curr.root - prev.root) % 12
    # Try both positive and negative forms to match INTERVAL_WEIGHTS
    neg_interval = raw_interval - 12

    weight = INTERVAL_WEIGHTS.get(raw_interval,
             INTERVAL_WEIGHTS.get(neg_interval, INTERVAL_ELSE))

    score = weight

    # Bonus for resolving to tonic
    if curr == tonic_chord:
        score += BONUS_TARGET_RESOLUTION

    # Penalty for non-diatonic chord
    if curr.chord_type == "INTRUDER":
        score += PENALTY_NON_DIATONIC

    # Penalty for unresolved dominant (intruder didn't go to tonic)
    if prev.chord_type == "INTRUDER" and curr != tonic_chord:
        score += PENALTY_UNRESOLVED_DOM

    return score


def harmonize_melody(
    melody_bars: list[BarData],
    root_name: str,
    mode_name: str,
) -> list[Chord]:
    """Viterbi decoder: find optimal chord sequence for the melody bars."""
    if not melody_bars:
        return []

    pool, tonic_chord = get_chord_pool(root_name, mode_name)

    # Build trellis: list of dicts mapping Chord -> {"score", "parent"}
    trellis: list[dict[Chord, dict]] = []

    # Bar 0 — initialization
    bar0 = melody_bars[0]
    candidates0 = get_candidates_for_bar(
        bar0.beat1_pc, bar0.beat3_pc, pool, bar0.all_pcs
    )
    layer0: dict[Chord, dict] = {}
    for c in candidates0:
        bonus = 10 if c == tonic_chord else 0
        layer0[c] = {"score": bonus, "parent": None}
    trellis.append(layer0)

    # Forward pass
    for t in range(1, len(melody_bars)):
        bar = melody_bars[t]
        candidates = get_candidates_for_bar(
            bar.beat1_pc, bar.beat3_pc, pool, bar.all_pcs
        )
        layer: dict[Chord, dict] = {}
        prev_layer = trellis[t - 1]

        for curr in candidates:
            best_score = float("-inf")
            best_parent: Chord | None = None
            for prev, prev_data in prev_layer.items():
                s = prev_data["score"] + calculate_transition_score(prev, curr, tonic_chord)
                if s > best_score:
                    best_score = s
                    best_parent = prev
            layer[curr] = {"score": best_score, "parent": best_parent}

        trellis.append(layer)

    # Backward pass — trace back from best final chord
    result: list[Chord] = []
    best_final: Chord | None = None
    best_final_score = float("-inf")
    for chord, data in trellis[-1].items():
        if data["score"] > best_final_score:
            best_final_score = data["score"]
            best_final = chord

    if best_final is None:
        # Fallback: tonic for every bar
        return [tonic_chord] * len(melody_bars)

    # Trace back
    current = best_final
    for t in range(len(trellis) - 1, -1, -1):
        result.append(current)
        parent = trellis[t][current]["parent"]
        if parent is not None:
            current = parent
    result.reverse()

    return result


# ──────────────────────────────────────────────
#  Top-level integration
# ──────────────────────────────────────────────

def _compute_voicing(
    melody_midi: int,
    chord_pcs: frozenset,
    prev_voicing: list[int] | None = None
) -> list[int]:
    """
    Voice a chord below the melody.
    If prev_voicing is provided, choose notes connected nicely (voice leading)
    to the previous voicing. Otherwise, use a default open structure.
    """
    
    # 1. Default (Stateless) Strategy: Open Spread
    # Used for the first chord or if we want to reset
    if prev_voicing is None:
        melody_pc = melody_midi % 12
        
        def _pc_distance(a: int, b: int) -> int:
            d = abs(a - b) % 12
            return min(d, 12 - d)

        # Prioritize chord tones distant from the melody PC
        if melody_pc in chord_pcs:
             acc_pcs = [pc for pc in chord_pcs if pc != melody_pc]
        else:
             # Sort by distance to melody PC (closest first)
             by_dist = sorted(chord_pcs, key=lambda pc: _pc_distance(pc, melody_pc))
             # Skip the very closest one to avoid clustering too tight? 
             # Or just use them. The original code skipped index 0 (closest) usually?
             # Original: acc_pcs = list(by_dist[1:])
             # Let's keep original logic for consistency on start
             acc_pcs = list(by_dist[1:])

        # Sort remaining PCs by distance to melody PC descending (furthest first)?
        # Original: acc_pcs.sort(key=..., reverse=True)
        acc_pcs.sort(key=lambda pc: _pc_distance(pc, melody_pc), reverse=True)

        targets = [melody_midi - 14, melody_midi - 9, melody_midi - 4]
        floor = max(48, melody_midi - 19)  # never below C3
        voices: list[int] = []
        for target, pc in zip(targets, acc_pcs):
            note = pc
            while note < target - 5:
                note += 12
            alt = note - 12
            if alt >= floor and abs(alt - target) < abs(note - target):
                note = alt
            while note >= melody_midi:
                note -= 12
            while note < floor:
                note += 12
            voices.append(note)
        return sorted(voices)

    # 2. Stateful Strategy: Minimal Movement (Voice Leading)
    # Assign each voice to a distinct chord pitch class via greedy matching.
    floor = max(48, melody_midi - 19)
    sorted_prev = sorted(prev_voicing)

    # Build all (voice_idx, pc, candidate, distance) options
    options: list[tuple[int, int, int, int]] = []
    for vi, old_note in enumerate(sorted_prev):
        for pc in chord_pcs:
            diff = (pc - old_note) % 12
            if diff > 6:
                diff -= 12
            candidate = old_note + diff
            while candidate >= melody_midi:
                candidate -= 12
            while candidate < floor:
                candidate += 12
            if candidate >= melody_midi:
                continue
            dist = abs(candidate - old_note)
            options.append((vi, pc, candidate, dist))

    # Greedy assignment: pick smallest distance first, no duplicate PCs
    options.sort(key=lambda x: x[3])
    used_pcs: set[int] = set()
    assigned: dict[int, int] = {}
    for vi, pc, candidate, dist in options:
        if vi in assigned or pc in used_pcs:
            continue
        assigned[vi] = candidate
        used_pcs.add(pc)

    # If voice leading produced fewer than 3 voices, fill from unused PCs
    unused_pcs = [pc for pc in chord_pcs if pc not in used_pcs]
    for pc in unused_pcs:
        if len(assigned) >= 3:
            break
        # Place near the middle of the existing voicing
        if assigned:
            target = sum(assigned.values()) // len(assigned)
        else:
            target = melody_midi - 9
        note = pc
        while note < target - 6:
            note += 12
        while note >= melody_midi:
            note -= 12
        while note < floor:
            note += 12
        if note < melody_midi:
            idx = max(assigned.keys()) + 1 if assigned else 0
            assigned[idx] = note

    return sorted(assigned.values())


def build_jazz_chord_track(
    notes: list[int],
    durations: list[float],
    root_name: str,
    mode_name: str,
) -> list[tuple[float, list[int], str, float]]:
    """Build a chord track using the Viterbi harmonizer.

    Returns data in the same format as build_chord_track():
        (beat_time, voicing_midi_list, chord_name, chord_duration)
    """
    bars = extract_bars(notes, durations)
    if not bars:
        return []

    chords = harmonize_melody(bars, root_name, mode_name)

    result: list[tuple[float, list[int], str, float]] = []
    prev_voicing: list[int] | None = None

    for bar, chord in zip(bars, chords):
        # Find the melody note at bar start for voicing placement
        beats = _beat_times(durations)
        melody_note = notes[0]  # fallback
        for idx in range(len(notes)):
            if beats[idx] <= bar.start_beat + 0.01:
                melody_note = notes[idx]

        voicing = _compute_voicing(melody_note, chord.notes, prev_voicing)
        
        # Fallback if voicing failed (e.g. empty set) -> try stateless
        if not voicing:
            voicing = _compute_voicing(melody_note, chord.notes, None)
            
        result.append((bar.start_beat, voicing, chord.name, bar.duration))
        prev_voicing = voicing

    return result
