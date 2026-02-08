"""
Data Sonification Web App
=========================
Converts numeric data into a MIDI melody with optional jazz-chord
harmonization, in-browser playback, music sheet notation, and
per-note duration editing via keyboard/click.

Tech: Python 3.x · Streamlit · Pandas · midiutil · NumPy · VexFlow (JS)

Run:  streamlit run sonification_app.py
"""

from __future__ import annotations

import io
import os
import wave as wave_module

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
from midiutil import MIDIFile

from jazz_harmonizer import (
    NOTE_NAMES,
    MODAL_DEFINITIONS,
    quantize_to_mode,
    build_jazz_chord_track,
)

# ──────────────────────────────────────────────
#  Music Theorist – melody constants & logic
# ──────────────────────────────────────────────

MIDI_MIN = 55  # G3
MIDI_MAX = 88  # E6

SCALE_INTERVALS: dict[str, list[int]] = {
    "Cromatica": list(range(12)),
    "Do Maggiore": [0, 2, 4, 5, 7, 9, 11],
}

DEFAULT_DURATION = 0.5  # eighth-note in quarter-note beats

# ──────────────────────────────────────────────
#  Sample datasets
# ──────────────────────────────────────────────

SAMPLE_DATASETS: dict[str, dict] = {
    "Caduta Libera (Accelerazione)": {
        "description": "La distanza cresce quadraticamente col tempo (d = \u00bdgt\u00b2). "
                       "Il tono parte basso e sale drammaticamente.",
        "X": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "Y": [0,5,20,45,80,125,180,245,320,405,500,605,720,845,980,
              1125,1280,1445,1620,1805,2000],
        "x_label": "Tempo_s",
        "y_label": "Distanza_m",
    },
    "Moto Parabolico (Parabola)": {
        "description": "Una palla di cannone sparata verso l\u2019alto. "
                       "Tono in salita \u2192 apice \u2192 tono in discesa.",
        "X": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        "Y": [0,18,34,48,60,70,78,84,88,90,90,88,84,78,70,60,48,34,18,0],
        "x_label": "Tempo_s",
        "y_label": "Altezza_m",
    },
    "Legge di Raffreddamento di Newton": {
        "description": "Caff\u00e8 caldo che si raffredda. Calo brusco iniziale, "
                       "poi appiattimento graduale (decadimento esponenziale).",
        "X": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "Y": [90,75,63,54,46,40,35,31,28,25,23,22,21,20,19.5,19,
              18.5,18.2,18,17.9,17.8],
        "x_label": "Minuto",
        "y_label": "Temp_C",
    },
    "Moto Armonico Smorzato (Bungee Jump)": {
        "description": "Oscillazioni che si riducono nel tempo, convergendo su una nota centrale.",
        "X": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "Y": [100,80,20,-50,-80,-60,-10,40,60,40,10,-20,-40,-30,-5,
              15,25,20,5,-10,-15],
        "x_label": "Tempo_s",
        "y_label": "Spostamento",
    },
    "Effetto Doppler": {
        "description": "Variazione di frequenza di un clacson che ti passa accanto. "
                       "Tono alto costante \u2192 calo improvviso \u2192 tono basso costante.",
        "X": [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10],
        "Y": [550,548,545,540,530,510,480,400,300,200,150,110,90,80,
              75,72,70,68,67,66,65],
        "x_label": "Posizione",
        "y_label": "Frequenza_Hz",
    },
}


def _build_legal_notes(scale_name: str) -> list[int]:
    intervals = SCALE_INTERVALS[scale_name]
    return [n for n in range(MIDI_MIN, MIDI_MAX + 1) if (n % 12) in intervals]


def _quantize(raw_midi: float, legal_notes: list[int]) -> int:
    return min(legal_notes, key=lambda n: abs(n - raw_midi))


def map_to_midi(values: pd.Series, scale_name: str) -> list[int]:
    legal_notes = _build_legal_notes(scale_name)
    v_min, v_max = float(values.min()), float(values.max())
    if v_min == v_max:
        mid = (MIDI_MIN + MIDI_MAX) / 2.0
        return [_quantize(mid, legal_notes)] * len(values)
    raw = MIDI_MIN + (values - v_min) / (v_max - v_min) * (MIDI_MAX - MIDI_MIN)
    return [_quantize(float(r), legal_notes) for r in raw]


# ──────────────────────────────────────────────
#  Music Theorist – chord harmonization
# ──────────────────────────────────────────────

C_MAJOR_PC_SET = frozenset({0, 2, 4, 5, 7, 9, 11})

DIATONIC_7THS: list[tuple[str, frozenset[int]]] = [
    ("Cmaj7",  frozenset({0, 4, 7, 11})),
    ("Dm7",    frozenset({0, 2, 5, 9})),
    ("Em7",    frozenset({2, 4, 7, 11})),
    ("Fmaj7",  frozenset({0, 4, 5, 9})),
    ("G7",     frozenset({2, 5, 7, 11})),
    ("Am7",    frozenset({0, 4, 7, 9})),
    ("Bm7b5",  frozenset({2, 5, 9, 11})),
]

CHORD_EVERY_N = 4
CHORD_INSTRUMENT = 48
CHORD_CHANNEL = 1
CHORD_VELOCITY = 100


def _pc_distance(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def _snap_to_c_major_pc(pc: int) -> int:
    return min(C_MAJOR_PC_SET, key=lambda p: _pc_distance(p, pc))


def _chords_containing_pc(pc: int) -> list[tuple[str, frozenset[int]]]:
    return [(n, pcs) for n, pcs in DIATONIC_7THS if pc in pcs]


def _compute_voicing(melody_midi: int, chord_pcs: frozenset[int]) -> list[int]:
    melody_pc = melody_midi % 12
    if melody_pc in chord_pcs:
        acc_pcs = [pc for pc in chord_pcs if pc != melody_pc]
    else:
        by_dist = sorted(chord_pcs, key=lambda pc: _pc_distance(pc, melody_pc))
        acc_pcs = list(by_dist[1:])
    acc_pcs.sort(key=lambda pc: _pc_distance(pc, melody_pc), reverse=True)
    targets = [melody_midi - 19, melody_midi - 12, melody_midi - 5]
    voices: list[int] = []
    for target, pc in zip(targets, acc_pcs):
        note = pc
        while note < target - 5:
            note += 12
        alt = note - 12
        if alt >= 36 and abs(alt - target) < abs(note - target):
            note = alt
        while note >= melody_midi:
            note -= 12
        while note < 36:
            note += 12
        voices.append(note)
    return sorted(voices)


def _voice_leading_cost(a: list[int], b: list[int]) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def _beat_times(durations: list[float]) -> list[float]:
    times: list[float] = []
    t = 0.0
    for d in durations:
        times.append(t)
        t += d
    return times


def build_chord_track(
    notes: list[int],
    durations: list[float],
) -> list[tuple[float, list[int], str, float]]:
    beats = _beat_times(durations)
    chords: list[tuple[float, list[int], str, float]] = []
    prev_voicing: list[int] | None = None

    for i in range(0, len(notes), CHORD_EVERY_N):
        beat_time = beats[i]
        chord_dur = sum(durations[i : i + CHORD_EVERY_N])
        melody_note = notes[i]
        melody_pc = melody_note % 12
        lookup_pc = (
            melody_pc if melody_pc in C_MAJOR_PC_SET
            else _snap_to_c_major_pc(melody_pc)
        )
        candidates = _chords_containing_pc(lookup_pc)
        if not candidates:
            candidates = [DIATONIC_7THS[0]]

        best_voicing: list[int] | None = None
        best_name = candidates[0][0]
        best_cost = float("inf")

        for name, pcs in candidates:
            voicing = _compute_voicing(melody_note, pcs)
            if prev_voicing is None:
                best_voicing = voicing
                best_name = name
                break
            cost = _voice_leading_cost(prev_voicing, voicing)
            if cost < best_cost:
                best_cost = cost
                best_voicing = voicing
                best_name = name

        assert best_voicing is not None
        chords.append((beat_time, best_voicing, best_name, chord_dur))
        prev_voicing = best_voicing

    return chords


# ──────────────────────────────────────────────
#  Data Engineer – parsing & cleaning
# ──────────────────────────────────────────────

def parse_input(uploaded_file=None, pasted_text: str = "") -> pd.DataFrame | None:
    df: pd.DataFrame | None = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("Impossibile leggere il file CSV caricato.")
            return None
    elif pasted_text.strip():
        text = pasted_text.strip()
        for sep in (",", "\t"):
            try:
                candidate = pd.read_csv(io.StringIO(text), sep=sep)
                if candidate.shape[1] >= 2:
                    df = candidate
                    break
            except Exception:
                continue
        if df is None:
            st.error(
                "Impossibile interpretare il testo incollato. "
                "Assicurati che ci siano almeno due colonne separate da virgole o tabulazioni."
            )
            return None

    if df is None or df.shape[1] < 2:
        return None

    df = df.iloc[:, :2].copy()
    df.columns = ["X", "Y"]
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"]).reset_index(drop=True)

    if df.empty:
        st.warning("Nessun valore numerico valido trovato nella seconda colonna.")
        return None
    return df


# ──────────────────────────────────────────────
#  MIDI generation
# ──────────────────────────────────────────────

def generate_midi(
    notes: list[int],
    durations: list[float],
    tempo: int,
    chord_track: list[tuple[float, list[int], str, float]] | None = None,
    melody_instrument: str = "Piano",
    harmony_instrument: str = "Pad",
) -> bytes:
    num_tracks = 2 if chord_track else 1
    midi = MIDIFile(num_tracks)

    mel_program = INSTRUMENT_DEFS[melody_instrument]["midi_program"]
    midi.addTempo(0, 0, tempo)
    midi.addProgramChange(0, 0, 0, mel_program)
    beats = _beat_times(durations)
    for pitch, bt, dur in zip(notes, beats, durations):
        midi.addNote(0, 0, pitch, bt, dur, 100)

    if chord_track:
        harm_program = INSTRUMENT_DEFS[harmony_instrument]["midi_program"]
        midi.addTempo(1, 0, tempo)
        midi.addProgramChange(1, CHORD_CHANNEL, 0, harm_program)
        for beat_time, chord_notes, _name, chord_dur in chord_track:
            for pitch in chord_notes:
                midi.addNote(
                    1, CHORD_CHANNEL, pitch,
                    beat_time, chord_dur, CHORD_VELOCITY,
                )

    buf = io.BytesIO()
    midi.writeFile(buf)
    return buf.getvalue()


# ──────────────────────────────────────────────
#  Audio synthesis (WAV)
# ──────────────────────────────────────────────

import base64
from scipy.io import wavfile
import scipy.signal

# ──────────────────────────────────────────────
#  Audio synthesis (WAV)
# ──────────────────────────────────────────────

SAMPLE_RATE = 44100

# ──────────────────────────────────────────────
#  Instrument definitions & sample generation
# ──────────────────────────────────────────────

INSTRUMENT_DEFS: dict[str, dict] = {
    "Piano": {
        "file": "assets/piano_c4.wav",
        "base_midi": 60,
        "attack": 0.012,
        "release": 0.08,
        "gain": 0.9,
        "midi_program": 0,
    },
    "Pad": {
        "file": "assets/pad_c4.wav",
        "base_midi": 60,
        "attack": 0.3,
        "release": 0.6,
        "gain": 0.85,
        "midi_program": 89,
    },
    "Choir": {
        "file": "assets/choir_c4.wav",
        "base_midi": 60,
        "attack": 0.25,
        "release": 0.5,
        "gain": 0.85,
        "midi_program": 52,
    },
}


def _generate_piano_sample() -> None:
    """Additive synthesis piano at C4 (261.63 Hz), ~3s, normalised to 90%."""
    sr = SAMPLE_RATE
    dur = 3.0
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    freq = 261.63
    two_pi = 2.0 * np.pi
    B = 0.0004  # inharmonicity coefficient

    # 12 harmonics with natural decay
    amps = [1.0, 0.65, 0.40, 0.28, 0.18, 0.12, 0.08, 0.05, 0.035, 0.02, 0.015, 0.01]
    sig = np.zeros(n, dtype=np.float64)
    for h, amp in enumerate(amps, 1):
        f_h = freq * h * np.sqrt(1.0 + B * h * h)
        decay = 0.6 + h * 0.35
        sig += amp * np.sin(two_pi * f_h * t) * np.exp(-decay * t)

    # Attack noise transient
    noise_len = int(0.008 * sr)
    if noise_len > 0:
        noise = np.random.default_rng(42).normal(0, 0.15, noise_len)
        noise *= np.linspace(1.0, 0.0, noise_len)
        sig[:noise_len] += noise

    # Short attack ramp
    att = int(0.005 * sr)
    if att > 0:
        sig[:att] *= np.linspace(0.0, 1.0, att)

    # Normalise to 90%
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.9

    pcm = (sig * 32767).astype(np.int16)
    os.makedirs("assets", exist_ok=True)
    wavfile.write("assets/piano_c4.wav", sr, pcm)


def _generate_pad_sample() -> None:
    """8 detuned sawtooth oscillators, LP-filtered at 2500 Hz, ~5s, at C4."""
    sr = SAMPLE_RATE
    dur = 5.0
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    freq = 261.63
    two_pi = 2.0 * np.pi

    # 8 oscillators with slight detuning (built via additive harmonics)
    detunes = [-0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08]
    sig = np.zeros(n, dtype=np.float64)
    for dt in detunes:
        f = freq * (1.0 + dt / 100.0)  # cents-level detuning
        osc = np.zeros(n, dtype=np.float64)
        # Sawtooth via additive: sum of sin(k*f)/k for k=1..20
        for k in range(1, 21):
            osc += np.sin(two_pi * f * k * t) / k
        sig += osc

    sig /= len(detunes)

    # Low-pass filter at ~2500 Hz
    nyq = sr / 2.0
    cutoff = min(2500.0 / nyq, 0.99)
    b, a = scipy.signal.butter(4, cutoff, btype='low')
    sig = scipy.signal.lfilter(b, a, sig)

    # Slow ADSR envelope
    env = np.ones(n, dtype=np.float64)
    att_s = int(0.4 * sr)
    rel_s = int(0.8 * sr)
    if att_s > 0:
        env[:att_s] = np.linspace(0.0, 1.0, att_s)
    if rel_s > 0 and rel_s < n:
        env[-rel_s:] *= np.linspace(1.0, 0.0, rel_s)
    sig *= env

    # Normalise to 90%
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.9

    pcm = (sig * 32767).astype(np.int16)
    os.makedirs("assets", exist_ok=True)
    wavfile.write("assets/pad_c4.wav", sr, pcm)


def _generate_choir_sample() -> None:
    """Formant synthesis 'aah' choir at C4, 5 detuned sources with vibrato, ~5s."""
    sr = SAMPLE_RATE
    dur = 5.0
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    freq = 261.63
    two_pi = 2.0 * np.pi

    # 5 detuned harmonic sources with vibrato
    detunes = [-0.06, -0.03, 0.0, 0.03, 0.06]
    sig = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(77)
    for dt in detunes:
        f = freq * (1.0 + dt / 100.0)
        # Per-voice vibrato: ~5 Hz, depth ~4 cents
        vib_rate = 4.5 + rng.uniform(-0.5, 0.5)
        vib_depth = f * 0.0025
        vib = vib_depth * np.sin(two_pi * vib_rate * t)
        # Build harmonic-rich source (20 harmonics)
        osc = np.zeros(n, dtype=np.float64)
        for k in range(1, 21):
            osc += np.sin(two_pi * (f + vib) * k * t) / (k ** 0.8)
        sig += osc

    sig /= len(detunes)

    # "Aah" vowel formant filtering (F1:730, F2:1090, F3:2440 Hz)
    nyq = sr / 2.0
    formants = [(730, 80), (1090, 90), (2440, 120)]  # (center_freq, bandwidth)
    filtered = np.zeros(n, dtype=np.float64)
    for fc, bw in formants:
        low = max((fc - bw / 2) / nyq, 0.001)
        high = min((fc + bw / 2) / nyq, 0.999)
        if low >= high:
            continue
        b, a = scipy.signal.butter(2, [low, high], btype='band')
        filtered += scipy.signal.lfilter(b, a, sig) * 1.5

    # Low-pass at 4 kHz
    lp_cut = min(4000.0 / nyq, 0.99)
    b, a = scipy.signal.butter(3, lp_cut, btype='low')
    filtered = scipy.signal.lfilter(b, a, filtered)

    # ADSR envelope
    env = np.ones(n, dtype=np.float64)
    att_s = int(0.35 * sr)
    rel_s = int(0.6 * sr)
    if att_s > 0:
        env[:att_s] = np.linspace(0.0, 1.0, att_s)
    if rel_s > 0 and rel_s < n:
        env[-rel_s:] *= np.linspace(1.0, 0.0, rel_s)
    filtered *= env

    # Normalise to 90%
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered / peak * 0.9

    pcm = (filtered * 32767).astype(np.int16)
    os.makedirs("assets", exist_ok=True)
    wavfile.write("assets/choir_c4.wav", sr, pcm)


def ensure_samples() -> None:
    """Generate any missing instrument sample files."""
    generators = {
        "assets/piano_c4.wav": _generate_piano_sample,
        "assets/pad_c4.wav": _generate_pad_sample,
        "assets/choir_c4.wav": _generate_choir_sample,
    }
    for path, gen_fn in generators.items():
        if not os.path.exists(path):
            gen_fn()

class Sampler:
    def __init__(self, sample_path: str, base_midi: int = 60,
                 attack: float = 0.1, release: float = 0.5, gain: float = 1.0):
        self.sample_path = sample_path
        self.base_midi = base_midi
        self.attack = attack
        self.release = release
        self.gain = gain
        self.sr = SAMPLE_RATE
        self.data = np.array([], dtype=np.float32)
        try:
            sr, data = wavfile.read(sample_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32767.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483647.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            if sr != SAMPLE_RATE:
                num_samples = int(len(data) * SAMPLE_RATE / sr)
                data = scipy.signal.resample(data, num_samples)

            self.data = data
        except Exception as e:
            print(f"Error loading sample {sample_path}: {e}")

    def get_sample(self, midi_note: int, duration_sec: float) -> np.ndarray:
        if len(self.data) == 0:
            return np.zeros(int(duration_sec * SAMPLE_RATE))

        semitones = midi_note - self.base_midi
        speed = 2.0 ** (semitones / 12.0)

        target_len_samples = int(len(self.data) / speed)
        old_indices = np.arange(0, len(self.data))
        new_indices = np.linspace(0, len(self.data) - 1, target_len_samples)
        resampled = np.interp(new_indices, old_indices, self.data)

        req_samples = int(duration_sec * SAMPLE_RATE)

        # ADSR envelope using per-instrument attack/release
        env = np.ones(req_samples, dtype=np.float32)
        att = int(self.attack * SAMPLE_RATE)
        rel = int(self.release * SAMPLE_RATE)

        if req_samples > 0:
            if att > 0:
                actual_att = min(att, req_samples)
                env[:actual_att] = np.linspace(0.0, 1.0, actual_att)
            if rel > 0:
                actual_rel = min(rel, req_samples)
                env[-actual_rel:] *= np.linspace(1.0, 0.0, actual_rel)

        if len(resampled) >= req_samples:
            output = resampled[:req_samples]
        else:
            output = np.zeros(req_samples, dtype=np.float32)
            ptr = 0
            while ptr < req_samples:
                chunk = min(len(resampled), req_samples - ptr)
                output[ptr:ptr + chunk] = resampled[:chunk]
                ptr += chunk

        return output * env * self.gain


def _apply_reverb(audio: np.ndarray, decay: float = 0.5) -> np.ndarray:
    processed = audio.copy()
    delays = [
        int(SAMPLE_RATE * 0.0297), 
        int(SAMPLE_RATE * 0.0371), 
        int(SAMPLE_RATE * 0.0411), 
        int(SAMPLE_RATE * 0.0437), 
    ]
    for d in delays:
        if d >= len(processed): continue
        processed[d:] += processed[:-d] * decay
        d2 = d * 2
        if d2 < len(processed):
            processed[d2:] += processed[:-d2] * (decay * 0.5)
    return processed * 0.6


def _get_sampler(instrument_name: str) -> Sampler:
    """Create a Sampler from an INSTRUMENT_DEFS entry."""
    d = INSTRUMENT_DEFS[instrument_name]
    return Sampler(
        sample_path=d["file"],
        base_midi=d["base_midi"],
        attack=d["attack"],
        release=d["release"],
        gain=d["gain"],
    )


def synthesize_wav(
    notes: list[int],
    durations: list[float],
    tempo: int,
    chord_track: list[tuple[float, list[int], str, float]] | None = None,
    melody_instrument: str = "Piano",
    harmony_instrument: str = "Pad",
) -> bytes:
    beat_sec = 60.0 / tempo
    beats = _beat_times(durations)

    total_sec_melody = (beats[-1] + durations[-1]) * beat_sec if notes else 0.0
    melody_total = int(SAMPLE_RATE * total_sec_melody)

    if chord_track:
        last_bt, _, _, last_cd = chord_track[-1]
        last_end_sec = (last_bt + last_cd) * beat_sec
        total_samples = max(melody_total, int(last_end_sec * SAMPLE_RATE) + SAMPLE_RATE)
    else:
        total_samples = melody_total + SAMPLE_RATE // 2

    if total_samples == 0:
        total_samples = 1

    audio_melody = np.zeros(total_samples, dtype=np.float64)
    audio_chords = np.zeros(total_samples, dtype=np.float64)

    # 1. Render Melody using Sampler
    mel_sampler = _get_sampler(melody_instrument)
    for pitch, bt, dur in zip(notes, beats, durations):
        note_sec = dur * beat_sec
        spn = int(SAMPLE_RATE * note_sec)
        if spn == 0:
            continue
        sig = mel_sampler.get_sample(pitch, note_sec)
        start = int(bt * beat_sec * SAMPLE_RATE)
        end = min(start + len(sig), total_samples)
        audio_melody[start:end] += sig[: end - start]

    # 2. Render Chords using Sampler
    if chord_track:
        harm_sampler = _get_sampler(harmony_instrument)
        for beat_time, chord_notes, _name, chord_dur in chord_track:
            chord_sec = chord_dur * beat_sec
            spc = int(SAMPLE_RATE * chord_sec)
            if spc == 0:
                continue
            start = int(beat_time * beat_sec * SAMPLE_RATE)

            chord_sig = np.zeros(spc, dtype=np.float64)
            for pitch in chord_notes:
                s = harm_sampler.get_sample(pitch, chord_sec)
                chunk = min(len(s), spc)
                chord_sig[:chunk] += s[:chunk]

            end = min(start + spc, total_samples)
            audio_chords[start:end] += chord_sig[: end - start]

    # 3. Normalise melody and chords independently before mixing
    mel_peak = np.max(np.abs(audio_melody))
    if mel_peak > 0:
        audio_melody /= mel_peak

    chd_peak = np.max(np.abs(audio_chords))
    if chd_peak > 0:
        audio_chords /= chd_peak

    # 4. Apply Reverb to Chords
    audio_chords_reverb = _apply_reverb(audio_chords, decay=0.5)

    # 5. Mix — both clearly audible
    final_mix = audio_melody * 0.55 + audio_chords_reverb * 0.45

    peak = np.max(np.abs(final_mix))
    if peak > 0:
        final_mix = final_mix / peak * 0.9

    pcm = (final_mix * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave_module.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ──────────────────────────────────────────────
#  Notation renderer (single-line scrolling + playhead)
# ──────────────────────────────────────────────

import json as _json


def render_notation_html(
    notes: list[int],
    durations: list[float],
    chord_data_js: list[dict] | None,
    tempo: int,
    melody_instrument: str = "Piano",
    harmony_instrument: str = "Pad",
) -> None:
    """Single-line VexFlow stave with synced playhead and interactive editing."""
    if not notes:
        return

    VF_NOTE = [
        ("c", None), ("c", "#"), ("d", None), ("e", "b"),
        ("e", None), ("f", None), ("f", "#"), ("g", None),
        ("a", "b"),  ("a", None), ("b", "b"), ("b", None),
    ]
    NAMES = ["Do","Do#","Re","Mib","Mi","Fa","Fa#","Sol","Lab","La","Sib","Si"]

    js_notes = []
    for midi, dur in zip(notes, durations):
        pc = midi % 12
        octave = midi // 12 - 1
        letter, acc = VF_NOTE[pc]
        js_notes.append({
            "midi": midi, "key": f"{letter}/{octave}",
            "acc": acc, "dur": dur, "name": f"{NAMES[pc]}{octave}",
        })

    total_beats = sum(durations)
    stave_w = max(900, int(total_beats * 110) + 200)

    notes_json = _json.dumps(js_notes)
    chords_json = _json.dumps(chord_data_js or [])

    # Load TWO samples: one for melody, one for harmony
    def _load_sample_b64(instrument_name: str) -> str:
        path = INSTRUMENT_DEFS[instrument_name]["file"]
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except Exception:
            return ""

    mel_b64 = _load_sample_b64(melody_instrument)
    harm_b64 = _load_sample_b64(harmony_instrument)

    # Envelope params for JS playback
    mel_def = INSTRUMENT_DEFS[melody_instrument]
    harm_def = INSTRUMENT_DEFS[harmony_instrument]

    html = f"""\
<!DOCTYPE html><html><head>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent}}
#wrap{{outline:none}}
#bar{{display:flex;align-items:center;gap:10px;padding:7px 10px;background:#f0f2f6;border-radius:6px;margin-bottom:5px;flex-wrap:wrap}}
#bar button{{padding:5px 14px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer;font-size:13px}}
#bar button:hover{{background:#e8e8e8}}
#bar button:disabled{{opacity:.45;cursor:default}}
#info{{font-weight:600;font-size:13px;color:#333}}
.hint{{color:#999;font-size:11px;margin-left:auto}}
#sw{{overflow-x:auto;overflow-y:hidden;position:relative;background:#fff;border-radius:8px;border:1px solid #eee;cursor:pointer}}
#vc{{position:relative;display:inline-block}}
#ph{{position:absolute;top:0;width:2.5px;background:#e63946;display:none;z-index:10;pointer-events:none;border-radius:1px;opacity:.85}}
</style></head><body>
<div id="wrap" tabindex="0">
<div id="bar">
  <button id="pbtn" onclick="doPlay()">&#9654; Riproduci</button>
  <button onclick="doStop()">&#9632; Ferma</button>
  <span id="info">Seleziona una nota</span>
  <span class="hint">&#8592;&#8594; Seleziona &nbsp;|&nbsp; &#8593;&#8595; Durata &nbsp;|&nbsp; Spazio = play/stop</span>
</div>
<div id="sw"><div id="vc"><div id="ph"></div></div></div>
<div id="dbg" style="color:#555;font-size:11px;margin-top:5px"></div>
</div>
<script>
(function(){{
/* ── data from Python ── */
var N={notes_json},CH={chords_json},TEMPO={tempo},BS=60/TEMPO,SW={stave_w};
var MEL_B64="{mel_b64}";
var HARM_B64="{harm_b64}";
var MEL_BUF=null,HARM_BUF=null;
var MEL_ATT={mel_def['attack']},MEL_REL={mel_def['release']};
var HARM_ATT={harm_def['attack']},HARM_REL={harm_def['release']};

function log(msg){{
  var d=document.getElementById('dbg');
  if(d) d.textContent=msg;
  console.log(msg);
}}
log("Ready. Chords: "+CH.length);

/* ── constants ── */
var DS=[.25,.5,1,2,4],DL=['16esimo','8vo','Quarto','Met\\u00e0','Intero'];
var D2V={{'0.25':'16','0.5':'8','1':'q','2':'h','4':'w'}};
var NAMES=['Do','Do#','Re','Mib','Mi','Fa','Fa#','Sol','Lab','La','Sib','Si'];
/* ── mutable state ── */
var durs=N.map(function(n){{return n.dur}});
var sel=0,xpos=[],bt=[],totB=0;
var playing=false,actx=null,t0=0,totS=0,afid=null;

function compBt(){{bt=[];var t=0;for(var i=0;i<durs.length;i++){{bt.push(t);t+=durs[i]}}totB=t}}
function dVF(d){{return D2V[String(d)]||'8'}}
function dLab(d){{var i=DS.indexOf(d);return i>=0?DL[i]:'?'}}
function info(){{
  var e=document.getElementById('info');
  if(!N.length){{e.textContent='Nessuna nota';return}}
  e.textContent='Nota '+sel+' \\u2014 '+N[sel].name+' ('+dLab(durs[sel])+')'
}}

/* ── render ── */
function render(){{
  compBt();
  SW=Math.max(900,Math.ceil(totB*110)+200);
  var c=document.getElementById('vc');
  var old=c.querySelector('svg');if(old)old.remove();
  xpos=[];if(!N.length)return;
  c.style.width=SW+'px';
  var VF=Vex.Flow,H=280;
  var r=new VF.Renderer(c,VF.Renderer.Backends.SVG);
  r.resize(SW,H);var ctx=r.getContext();ctx.setFont('Arial',10,'');
  var st=new VF.Stave(10,80,SW-30);
  st.addClef('treble').addTimeSignature('4/4');
  st.setContext(ctx).draw();
  /* chord lookup */
  var chM={{}};for(var i=0;i<CH.length;i++)chM[CH[i].beat.toFixed(4)]=CH[i].name;
  var vn=[];
  for(var i=0;i<N.length;i++){{
    var sn=new VF.StaveNote({{clef:'treble',keys:[N[i].key],duration:dVF(durs[i])}});
    if(N[i].acc)sn.addAccidental(0,new VF.Accidental(N[i].acc));
    if(i===sel)sn.setStyle({{fillStyle:'#DAA520',strokeStyle:'#DAA520'}});
    var cn=chM[bt[i].toFixed(4)]||null;
    if(cn)sn.addModifier(0,new VF.Annotation(cn).setVerticalJustification(VF.Annotation.VerticalJustify.TOP).setFont('Arial',12,'bold'));
    vn.push(sn)
  }}
  var v=new VF.Voice({{num_beats:4,beat_value:4}});v.setMode(VF.Voice.Mode.SOFT);
  v.addTickables(vn);new VF.Formatter().joinVoices([v]).format([v],SW-100);
  v.draw(ctx,st);
  for(var i=0;i<vn.length;i++)xpos.push(vn[i].getAbsoluteX());
  try{{var bm=VF.Beam.generateBeams(vn,{{groups:[new VF.Fraction(2,8)]}});bm.forEach(function(b){{b.setContext(ctx).draw()}})}}catch(e){{}}
  /* ── barlines every 4 beats ── */
  var svg=c.querySelector('svg');
  var stTop=st.getYForLine(0),stBot=st.getYForLine(4);
  for(var b=4;b<totB;b+=4){{
    var bx=-1;
    for(var j=0;j<bt.length;j++){{if(bt[j]>=b-0.01){{bx=xpos[j]-15;break}}}}
    if(bx>0){{
      var ln=document.createElementNS('http://www.w3.org/2000/svg','line');
      ln.setAttribute('x1',bx);ln.setAttribute('y1',stTop);
      ln.setAttribute('x2',bx);ln.setAttribute('y2',stBot);
      ln.setAttribute('stroke','#000');ln.setAttribute('stroke-width','1.5');
      svg.appendChild(ln)
    }}
  }}
  /* ── beat labels below stave ── */
  var BLBL=['one','and','two','and','three','and','four','and'];
  var BPOS=[0,.5,1,1.5,2,2.5,3,3.5];
  var lblY=stBot+28;
  for(var i=0;i<N.length;i++){{
    var inM=((bt[i]%4)+4)%4;
    var lbl='';
    for(var k=0;k<BPOS.length;k++){{if(Math.abs(inM-BPOS[k])<0.02){{lbl=BLBL[k];break}}}}
    if(lbl){{
      var tx=document.createElementNS('http://www.w3.org/2000/svg','text');
      tx.setAttribute('x',xpos[i]);tx.setAttribute('y',lblY);
      tx.setAttribute('text-anchor','middle');
      tx.setAttribute('font-size','10');tx.setAttribute('font-family','Arial');
      tx.setAttribute('fill','#888');tx.textContent=lbl;
      svg.appendChild(tx)
    }}
  }}
  document.getElementById('ph').style.height=H+'px';
  info();
  /* scroll to selected if off-screen */
  var sw=document.getElementById('sw'),vw=sw.clientWidth,sx=xpos[sel];
  if(sx!==undefined&&(sx<sw.scrollLeft+40||sx>sw.scrollLeft+vw-40))sw.scrollLeft=sx-vw/2;
}}

/* ── playhead mapping ── */
function b2x(b){{
  if(!xpos.length)return 0;
  if(b<=bt[0])return xpos[0];
  for(var i=0;i<bt.length-1;i++){{
    if(b>=bt[i]&&b<bt[i+1]){{var f=(b-bt[i])/(bt[i+1]-bt[i]);return xpos[i]+f*(xpos[i+1]-xpos[i])}}
  }}
  return xpos[xpos.length-1]
}}

/* ── unified sample-based note scheduler ── */
function m2f(m){{return 440*Math.pow(2,(m-69)/12)}}
function schedNote(ctx,dest,buf,midiNote,t,dur,vol,att,rel){{
  if(!buf)return;
  var src=ctx.createBufferSource();
  src.buffer=buf;
  src.playbackRate.value=m2f(midiNote)/261.63;
  var g=ctx.createGain();
  g.gain.setValueAtTime(0,t);
  g.gain.linearRampToValueAtTime(vol,t+att);
  g.gain.setValueAtTime(vol,t+Math.max(dur-0.01,att));
  g.gain.linearRampToValueAtTime(0,t+dur+rel);
  src.connect(g);g.connect(dest);
  src.start(t);src.stop(t+dur+rel+0.1);
}}

/* ── decode base64 to ArrayBuffer ── */
function b64toAB(b64){{
  var s=atob(b64),ab=new ArrayBuffer(s.length),ia=new Uint8Array(ab);
  for(var i=0;i<s.length;i++)ia[i]=s.charCodeAt(i);
  return ab;
}}

window.doPlay=function(){{
  doStop();
  actx=new(window.AudioContext||window.webkitAudioContext)();
  if(actx.state==='suspended')actx.resume();

  var pending=0,failed=false;
  function tryStart(){{
    if(pending>0)return;
    if(failed)log("Sample decode error — playing without samples");
    t0=actx.currentTime+0.15;var t=t0;
    /* melody */
    for(var i=0;i<N.length;i++){{
      var ds=durs[i]*BS;
      schedNote(actx,actx.destination,MEL_BUF,N[i].midi,t,ds,0.5,MEL_ATT,MEL_REL);
      t+=ds;
    }}
    /* chords */
    for(var i=0;i<CH.length;i++){{
      var ct=t0+CH[i].beat*BS,cd=CH[i].dur*BS;
      if(CH[i].notes){{
        for(var j=0;j<CH[i].notes.length;j++){{
          schedNote(actx,actx.destination,HARM_BUF,CH[i].notes[j],ct,cd,0.35,HARM_ATT,HARM_REL);
        }}
      }}
    }}
    totS=t-t0;playing=true;
    document.getElementById('pbtn').disabled=true;
    anim();
  }}

  /* Decode both samples in parallel */
  function decodeSample(b64,cb){{
    if(!b64||b64.length===0){{cb(null);return}}
    pending++;
    try{{
      actx.decodeAudioData(b64toAB(b64),function(buf){{pending--;cb(buf)}},function(){{pending--;failed=true;cb(null)}});
    }}catch(e){{pending--;failed=true;cb(null)}}
  }}

  document.getElementById('pbtn').textContent="Caricamento...";
  if(!MEL_BUF&&MEL_B64){{
    decodeSample(MEL_B64,function(buf){{MEL_BUF=buf;document.getElementById('pbtn').textContent="\\u25b6 Riproduci";tryStart()}});
  }}
  if(!HARM_BUF&&HARM_B64){{
    decodeSample(HARM_B64,function(buf){{HARM_BUF=buf;tryStart()}});
  }}
  /* if both already decoded */
  if(MEL_BUF&&(HARM_BUF||!HARM_B64)){{
    document.getElementById('pbtn').textContent="\\u25b6 Riproduci";
    tryStart();
  }}
}};
window.doStop=function(){{
  playing=false;if(afid){{cancelAnimationFrame(afid);afid=null}}
  if(actx){{actx.close();actx=null}}
  document.getElementById('ph').style.display='none';
  document.getElementById('pbtn').disabled=false
}};
function anim(){{
  if(!playing||!actx)return;
  var el=actx.currentTime-t0;
  if(el<0){{afid=requestAnimationFrame(anim);return}}
  if(el>=totS+.3){{doStop();return}}
  var x=b2x(el/BS);
  var ph=document.getElementById('ph');ph.style.left=x+'px';ph.style.display='block';
  var sw=document.getElementById('sw'),vw=sw.clientWidth;
  sw.scrollLeft=Math.max(0,x-vw*.3);
  afid=requestAnimationFrame(anim)
}}

/* ── click ── */
document.getElementById('vc').addEventListener('click',function(e){{
  if(!xpos.length)return;
  var r=this.getBoundingClientRect();
  var cx=e.clientX-r.left+document.getElementById('sw').scrollLeft;
  var bi=0,bd=1e9;
  for(var i=0;i<xpos.length;i++){{var dx=Math.abs(cx-xpos[i]);if(dx<bd){{bd=dx;bi=i}}}}
  sel=bi;render();document.getElementById('wrap').focus()
}});

/* ── keyboard ── */
document.addEventListener('keydown',function(e){{
  if(!N.length)return;var h=false;
  if(e.key==='ArrowLeft'&&sel>0){{sel--;h=true}}
  else if(e.key==='ArrowRight'&&sel<N.length-1){{sel++;h=true}}
  else if(e.key==='ArrowUp'){{var ci=DS.indexOf(durs[sel]);if(ci>=0&&ci<DS.length-1){{durs[sel]=DS[ci+1];h=true}}}}
  else if(e.key==='ArrowDown'){{var ci=DS.indexOf(durs[sel]);if(ci>0){{durs[sel]=DS[ci-1];h=true}}}}
  else if(e.key===' '){{e.preventDefault();if(playing)doStop();else doPlay();return}}
  if(h){{e.preventDefault();render()}}
}});
document.getElementById('wrap').addEventListener('click',function(){{this.focus()}});

/* ── load VexFlow async ── */
var s=document.createElement('script');
s.src='https://cdn.jsdelivr.net/npm/vexflow@3.0.9/releases/vexflow-min.js';
s.onload=function(){{render()}};
s.onerror=function(){{document.getElementById('info').textContent='Errore: VexFlow non caricato'}};
document.head.appendChild(s);
}})();
</script></body></html>"""
    st_components.html(html, height=350, scrolling=False)


# ──────────────────────────────────────────────
#  Frontend Developer – Streamlit UI
# ──────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Sonificazione Dati", layout="centered")
    ensure_samples()
    st.title("Sonificazione Dati")
    st.caption("Converti una colonna di numeri in una melodia MIDI.")

    # ── 1. Input ──
    st.subheader("1 · Dati in Ingresso")
    tab_upload, tab_paste, tab_samples = st.tabs(
        ["Carica CSV", "Incolla Testo", "Dataset di Esempio"]
    )
    with tab_upload:
        uploaded = st.file_uploader("Scegli un file .csv", type=["csv"])
    with tab_paste:
        pasted = st.text_area(
            "Incolla dati separati da virgole o tabulazioni (due colonne, intestazione opzionale)",
            height=180,
            placeholder="x,y\n1,440\n2,523\n3,392\n...",
        )
    with tab_samples:
        sample_name = st.selectbox(
            "Scegli un dataset di esempio",
            ["(nessuno)"] + list(SAMPLE_DATASETS.keys()),
        )
        if sample_name != "(nessuno)":
            st.caption(SAMPLE_DATASETS[sample_name]["description"])

    # Build DataFrame — sample dataset takes priority when selected
    df: pd.DataFrame | None = None
    if sample_name != "(nessuno)":
        s = SAMPLE_DATASETS[sample_name]
        df = pd.DataFrame({"X": s["X"], "Y": s["Y"]})
    else:
        df = parse_input(uploaded_file=uploaded, pasted_text=pasted)

    if df is None:
        st.info("Carica un CSV, incolla dei dati o scegli un dataset di esempio per iniziare.")
        return

    st.success(f"{len(df)} punti dati caricati.")

    # ── 2. Chart ──
    st.subheader("2 · Anteprima Dati")
    chart_df = pd.DataFrame({"Y": df["Y"].values}, index=range(len(df)))
    st.line_chart(chart_df)

    # ── 3. Settings ──
    st.subheader("3 · Impostazioni Sonificazione")
    col1, col2, col3 = st.columns(3)
    with col1:
        root_options = ["Cromatica"] + list(NOTE_NAMES)
        root_choice = st.selectbox("Tonalità", root_options)
    with col2:
        mode_options = list(MODAL_DEFINITIONS.keys())
        mode_choice = st.selectbox(
            "Modo",
            mode_options,
            disabled=(root_choice == "Cromatica"),
        )
    with col3:
        tempo = st.slider("Tempo (BPM)", 60, 240, 120, 1)

    use_modal = root_choice != "Cromatica"
    harmonize = st.checkbox(
        "Armonizza (accordi jazz di 7a)",
        disabled=not use_modal,
        help=(
            "Aggiunge accordi diatonici di settima con armonizzatore Viterbi. "
            "Disponibile solo selezionando una tonalità (non Cromatica)."
        ),
    )
    if not use_modal:
        harmonize = False

    # ── Instrument selection ──
    inst_names = list(INSTRUMENT_DEFS.keys())
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        melody_instrument = st.selectbox(
            "Strumento Melodia", inst_names, index=inst_names.index("Piano"),
        )
    with col_i2:
        harmony_instrument = st.selectbox(
            "Strumento Armonia",
            inst_names,
            index=inst_names.index("Pad"),
        )

    # ── Start button (latched via session state) ──
    st.divider()
    if st.button("Avvia Sonificazione", type="primary", use_container_width=True):
        st.session_state.sonification_started = True

    if not st.session_state.get("sonification_started"):
        st.info("Configura le impostazioni e premi **Avvia Sonificazione** per generare la musica.")
        return

    # ── Process melody ──
    if use_modal:
        # Map raw values to MIDI range, then quantize to the modal scale
        v_min, v_max = float(df["Y"].min()), float(df["Y"].max())
        if v_min == v_max:
            mid = (MIDI_MIN + MIDI_MAX) / 2.0
            raw_midi = [mid] * len(df)
        else:
            raw_midi = [
                MIDI_MIN + (v - v_min) / (v_max - v_min) * (MIDI_MAX - MIDI_MIN)
                for v in df["Y"]
            ]
        notes = quantize_to_mode(raw_midi, root_choice, mode_choice)
    else:
        notes = map_to_midi(df["Y"], "Cromatica")
    n_notes = len(notes)

    # ── Per-note durations (session state) ──
    if "durations" not in st.session_state or len(st.session_state.durations) != n_notes:
        st.session_state.durations = [DEFAULT_DURATION] * n_notes
    durations: list[float] = list(st.session_state.durations)

    # Chord track for display
    if harmonize and use_modal:
        chord_data = build_jazz_chord_track(notes, durations, root_choice, mode_choice)
    elif harmonize:
        chord_data = build_chord_track(notes, durations)
    else:
        chord_data = None

    # ── 4. Music Sheet ──
    st.subheader("4 · Spartito")

    chord_data_js = None
    if chord_data:
        chord_data_js = [
            {"beat": round(bt, 4), "name": n, "dur": d, "notes": list(voicing)}
            for bt, voicing, n, d in chord_data
        ]

    render_notation_html(notes, durations, chord_data_js, tempo,
                         melody_instrument, harmony_instrument)

    # Rebuild chord track with final durations (for audio/MIDI)
    if harmonize and use_modal:
        chord_data_final = build_jazz_chord_track(notes, durations, root_choice, mode_choice)
    elif harmonize:
        chord_data_final = build_chord_track(notes, durations)
    else:
        chord_data_final = None

    # ── 5. Download ──
    st.subheader("5 · Scarica")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        midi_bytes = generate_midi(
            notes, durations, tempo, chord_data_final,
            melody_instrument=melody_instrument,
            harmony_instrument=harmony_instrument,
        )
        st.download_button(
            label="Scarica MIDI",
            data=midi_bytes,
            file_name="sonificazione.mid",
            mime="audio/midi",
        )
    with col_dl2:
        wav_bytes = synthesize_wav(
            notes, durations, tempo, chord_data_final,
            melody_instrument=melody_instrument,
            harmony_instrument=harmony_instrument,
        )
        st.download_button(
            label="Scarica WAV",
            data=wav_bytes,
            file_name="sonificazione.wav",
            mime="audio/wav",
        )

    with st.expander("Sequenza note MIDI"):
        st.code(" ".join(str(n) for n in notes), language=None)

    if chord_data_final:
        with st.expander("Voci degli accordi"):
            lines = []
            for beat, voicing, name, cdur in chord_data_final:
                v_str = ", ".join(str(n) for n in voicing)
                lines.append(f"Battuta {beat:5.1f}  {name:<7s}  [{v_str}]  dur={cdur:.2f}")
            st.code("\n".join(lines), language=None)


if __name__ == "__main__":
    main()
