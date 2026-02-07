"""
Data Sonification Web App
=========================
Converts numeric data into a MIDI melody with optional jazz-chord
harmonization, in-browser playback, music sheet notation, and
per-note duration editing.

Tech: Python 3.x · Streamlit · Pandas · midiutil · NumPy · VexFlow (JS)

Run:  streamlit run sonification_app.py
"""

from __future__ import annotations

import io
import json
import wave as wave_module

import numpy as np
import pandas as pd
import streamlit as st
from midiutil import MIDIFile

# ──────────────────────────────────────────────
#  Music Theorist – melody constants & logic
# ──────────────────────────────────────────────

MIDI_MIN = 55  # G3
MIDI_MAX = 88  # E6

SCALE_INTERVALS: dict[str, list[int]] = {
    "Chromatic": list(range(12)),
    "C Major":   [0, 2, 4, 5, 7, 9, 11],
}

# Duration helpers
DURATION_STEPS  = [0.25,  0.5,   1.0,     2.0,   4.0]
DURATION_LABELS = ["16th", "8th", "Quarter", "Half", "Whole"]
DURATION_TO_VF  = {0.25: "16", 0.5: "8", 1.0: "q", 2.0: "h", 4.0: "w"}
DEFAULT_DURATION = 0.5  # eighth-note in quarter-note beats


def _build_legal_notes(scale_name: str) -> list[int]:
    """Return every MIDI note in [MIDI_MIN, MIDI_MAX] that belongs to *scale_name*."""
    intervals = SCALE_INTERVALS[scale_name]
    return [n for n in range(MIDI_MIN, MIDI_MAX + 1) if (n % 12) in intervals]


def _quantize(raw_midi: float, legal_notes: list[int]) -> int:
    """Snap a raw (float) MIDI value to the nearest legal note."""
    return min(legal_notes, key=lambda n: abs(n - raw_midi))


def map_to_midi(values: pd.Series, scale_name: str) -> list[int]:
    """Linear interpolation → MIDI range, then quantise to scale."""
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
CHORD_INSTRUMENT = 48   # GM String Ensemble 1
CHORD_CHANNEL = 1
CHORD_VELOCITY = 100     # same loudness as melody


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
    """Cumulative beat positions from a list of per-note durations."""
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
    """
    Harmonize *notes* with diatonic 7th chords every CHORD_EVERY_N notes.

    Returns ``[(beat_time, [midi, midi, midi], chord_name, chord_dur), …]``.
    """
    beats = _beat_times(durations)
    chords: list[tuple[float, list[int], str, float]] = []
    prev_voicing: list[int] | None = None

    for i in range(0, len(notes), CHORD_EVERY_N):
        beat_time = beats[i]
        group_end = i + CHORD_EVERY_N
        chord_dur = sum(durations[i:group_end])

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
            st.error("Could not parse the uploaded CSV file.")
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
                "Could not parse the pasted text. "
                "Ensure at least two columns separated by commas or tabs."
            )
            return None

    if df is None or df.shape[1] < 2:
        return None

    df = df.iloc[:, :2].copy()
    df.columns = ["X", "Y"]
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"]).reset_index(drop=True)

    if df.empty:
        st.warning("No valid numeric values found in the second column.")
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
) -> bytes:
    num_tracks = 2 if chord_track else 1
    midi = MIDIFile(num_tracks)

    # Track 0 – melody (Piano)
    midi.addTempo(0, 0, tempo)
    midi.addProgramChange(0, 0, 0, 0)
    beats = _beat_times(durations)
    for pitch, bt, dur in zip(notes, beats, durations):
        midi.addNote(0, 0, pitch, bt, dur, 100)

    # Track 1 – chords (String Ensemble)
    if chord_track:
        midi.addTempo(1, 0, tempo)
        midi.addProgramChange(1, CHORD_CHANNEL, 0, CHORD_INSTRUMENT)
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

SAMPLE_RATE = 44100


def _midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _render_piano(freq: float, n_samples: int, duration_sec: float) -> np.ndarray:
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    two_pi = 2.0 * np.pi
    amps = (1.0, 0.72, 0.55, 0.38, 0.28, 0.18, 0.12, 0.08, 0.05, 0.03)
    B = 0.0004
    sig = np.zeros(n_samples, dtype=np.float64)
    for n, amp in enumerate(amps, 1):
        f_n = freq * n * np.sqrt(1.0 + B * n * n)
        decay = 1.8 + n * 0.7
        sig += amp * np.sin(two_pi * f_n * t) * np.exp(-decay * t)
    att = int(0.003 * SAMPLE_RATE)
    if 0 < att < n_samples:
        sig[:att] *= np.linspace(0.0, 1.0, att)
    rel = int(0.010 * SAMPLE_RATE)
    if 0 < rel < n_samples:
        sig[-rel:] *= np.linspace(1.0, 0.0, rel)
    return sig


def _render_pad(freq: float, n_samples: int, duration_sec: float) -> np.ndarray:
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    two_pi = 2.0 * np.pi
    amps = (0.70, 0.20, 0.08, 0.02)
    sig = np.zeros(n_samples, dtype=np.float64)
    for i, amp in enumerate(amps, 1):
        sig += amp * np.sin(two_pi * freq * i * t)
    env = np.ones(n_samples, dtype=np.float64)
    att = int(0.05 * SAMPLE_RATE)
    rel = int(0.10 * SAMPLE_RATE)
    if 0 < att < n_samples:
        env[:att] = np.linspace(0.0, 1.0, att)
    if 0 < rel < n_samples:
        env[-rel:] = np.linspace(1.0, 0.0, rel)
    return sig * env


def synthesize_wav(
    notes: list[int],
    durations: list[float],
    tempo: int,
    chord_track: list[tuple[float, list[int], str, float]] | None = None,
) -> bytes:
    beat_sec = 60.0 / tempo
    beats = _beat_times(durations)

    # Total melody length in samples
    total_sec_melody = (beats[-1] + durations[-1]) * beat_sec if notes else 0.0
    melody_total = int(SAMPLE_RATE * total_sec_melody)

    # Account for last chord ringing beyond melody
    if chord_track:
        last_bt, _, _, last_cd = chord_track[-1]
        last_end_sec = (last_bt + last_cd) * beat_sec
        total_samples = max(melody_total, int(last_end_sec * SAMPLE_RATE))
    else:
        total_samples = melody_total

    if total_samples == 0:
        total_samples = 1

    audio = np.zeros(total_samples, dtype=np.float64)

    # Melody (piano)
    for idx, (pitch, bt, dur) in enumerate(zip(notes, beats, durations)):
        note_sec = dur * beat_sec
        spn = int(SAMPLE_RATE * note_sec)
        if spn == 0:
            continue
        sig = _render_piano(_midi_to_freq(pitch), spn, note_sec)
        start = int(bt * beat_sec * SAMPLE_RATE)
        end = min(start + spn, total_samples)
        audio[start:end] += sig[: end - start]

    # Chords (string pad) — same loudness as melody
    if chord_track:
        for beat_time, chord_notes, _name, chord_dur in chord_track:
            chord_sec = chord_dur * beat_sec
            spc = int(SAMPLE_RATE * chord_sec)
            if spc == 0:
                continue
            start = int(beat_time * beat_sec * SAMPLE_RATE)
            for pitch in chord_notes:
                sig = _render_pad(_midi_to_freq(pitch), spc, chord_sec)
                end = min(start + spc, total_samples)
                audio[start:end] += sig[: end - start]

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave_module.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ──────────────────────────────────────────────
#  Music Notation – VexFlow rendering
# ──────────────────────────────────────────────

# Pitch-class → (letter, accidental|None)  for VexFlow note keys
_VF_NOTE_INFO = [
    ("c", None), ("c", "#"), ("d", None), ("e", "b"),
    ("e", None), ("f", None), ("f", "#"), ("g", None),
    ("a", "b"),  ("a", None), ("b", "b"), ("b", None),
]

_DISPLAY_NAMES = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
]

BEATS_PER_LINE = 8  # wrap notation every ~8 beats


def _midi_to_vf_key(midi_note: int) -> tuple[str, int, str | None]:
    """Return (letter, octave, accidental_or_None) for VexFlow."""
    pc = midi_note % 12
    octave = (midi_note // 12) - 1
    letter, acc = _VF_NOTE_INFO[pc]
    return letter, octave, acc


def _midi_to_note_name(midi_note: int) -> str:
    return _DISPLAY_NAMES[midi_note % 12] + str((midi_note // 12) - 1)


def render_notation_html(
    notes: list[int],
    durations: list[float],
    chord_track: list[tuple[float, list[int], str, float]] | None,
    selected_idx: int,
) -> str:
    """Build an HTML string that renders the score via VexFlow 3.0.9."""

    # Build chord-name lookup: beat_time → chord_name
    chord_at_beat: dict[float, str] = {}
    if chord_track:
        for bt, _, name, _ in chord_track:
            chord_at_beat[bt] = name

    beats = _beat_times(durations)

    # Build note data as JSON for JS
    note_data = []
    for i, (midi_n, dur, bt) in enumerate(zip(notes, durations, beats)):
        letter, octave, acc = _midi_to_vf_key(midi_n)
        vf_dur = DURATION_TO_VF.get(dur, "8")
        chord_name = chord_at_beat.get(bt)
        note_data.append({
            "keys": [f"{letter}/{octave}"],
            "duration": vf_dur,
            "accidental": acc,
            "selected": i == selected_idx,
            "chord": chord_name,
            "index": i,
        })

    # Split into lines of ~BEATS_PER_LINE
    lines: list[list[dict]] = []
    current_line: list[dict] = []
    line_beats = 0.0
    note_idx = 0
    for i, dur in enumerate(durations):
        if line_beats >= BEATS_PER_LINE and current_line:
            lines.append(current_line)
            current_line = []
            line_beats = 0.0
        current_line.append(note_data[i])
        line_beats += dur
    if current_line:
        lines.append(current_line)

    notes_json = json.dumps(lines)

    html = f"""
    <div id="vf-output" style="overflow-x:auto; background:#fff; padding:10px; border-radius:8px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/vexflow@3.0.9/releases/vexflow-min.js"></script>
    <script>
    (function() {{
        var VF = Vex.Flow;
        var container = document.getElementById('vf-output');
        container.innerHTML = '';

        var lines = {notes_json};
        var LINE_HEIGHT = 160;
        var totalHeight = lines.length * LINE_HEIGHT + 40;
        var WIDTH = Math.min(container.parentElement.clientWidth - 40, 900);

        var renderer = new VF.Renderer(container, VF.Renderer.Backends.SVG);
        renderer.resize(WIDTH, totalHeight);
        var context = renderer.getContext();
        context.setFont('Arial', 10, '');

        for (var li = 0; li < lines.length; li++) {{
            var lineNotes = lines[li];
            var stave = new VF.Stave(10, li * LINE_HEIGHT + 10, WIDTH - 30);
            if (li === 0) stave.addClef('treble');
            stave.setContext(context).draw();

            var vfNotes = [];
            for (var ni = 0; ni < lineNotes.length; ni++) {{
                var nd = lineNotes[ni];
                var sn = new VF.StaveNote({{
                    clef: 'treble',
                    keys: nd.keys,
                    duration: nd.duration
                }});

                if (nd.accidental) {{
                    sn.addAccidental(0, new VF.Accidental(nd.accidental));
                }}

                // Dot for dotted notes not used here but future-proof
                if (nd.selected) {{
                    sn.setStyle({{fillStyle: '#DAA520', strokeStyle: '#DAA520'}});
                }}

                // Chord name annotation above
                if (nd.chord) {{
                    sn.addModifier(0,
                        new VF.Annotation(nd.chord)
                            .setVerticalJustification(VF.Annotation.VerticalJustify.TOP)
                            .setFont('Arial', 11, 'bold')
                    );
                }}

                vfNotes.push(sn);
            }}

            var voice = new VF.Voice({{num_beats: 4, beat_value: 4}});
            voice.setMode(VF.Voice.Mode.SOFT);
            voice.addTickables(vfNotes);

            var formatter = new VF.Formatter();
            formatter.joinVoices([voice]).format([voice], WIDTH - 60);

            voice.draw(context, stave);

            // Beaming
            try {{
                var beams = VF.Beam.generateBeams(vfNotes, {{
                    groups: [new VF.Fraction(2, 8)]
                }});
                beams.forEach(function(b) {{ b.setContext(context).draw(); }});
            }} catch(e) {{}}
        }}
    }})();
    </script>
    """
    return html


# ──────────────────────────────────────────────
#  Frontend Developer – Streamlit UI
# ──────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Data Sonification", layout="centered")
    st.title("Data Sonification")
    st.caption("Convert a column of numbers into a MIDI melody.")

    # ── 1. Input ──
    st.subheader("1 · Input Data")
    tab_upload, tab_paste = st.tabs(["Upload CSV", "Paste Text"])
    with tab_upload:
        uploaded = st.file_uploader("Choose a .csv file", type=["csv"])
    with tab_paste:
        pasted = st.text_area(
            "Paste comma- or tab-separated data (two columns, header optional)",
            height=180,
            placeholder="x,y\n1,440\n2,523\n3,392\n...",
        )

    df = parse_input(uploaded_file=uploaded, pasted_text=pasted)
    if df is None:
        st.info("Upload a CSV or paste data above to get started.")
        return

    st.success(f"{len(df)} data points loaded.")

    # ── 2. Chart ──
    st.subheader("2 · Data Preview")
    chart_df = pd.DataFrame({"Y": df["Y"].values}, index=range(len(df)))
    st.line_chart(chart_df)

    # ── 3. Settings ──
    st.subheader("3 · Sonification Settings")
    col1, col2 = st.columns(2)
    with col1:
        scale = st.radio("Scale", list(SCALE_INTERVALS.keys()), horizontal=True)
    with col2:
        tempo = st.slider("Tempo (BPM)", 60, 240, 120, 1)

    harmonize = st.checkbox(
        "Harmonize (jazz 7th chords every 4 notes)",
        help=(
            "Adds diatonic 7th chords (C Major tertian harmony) on a "
            "separate Strings track.  Voicings use open spread with the "
            "melody on top and smooth voice-leading between changes."
        ),
    )

    # ── Process melody ──
    notes = map_to_midi(df["Y"], scale)
    n_notes = len(notes)

    # ── Per-note durations in session state ──
    if "durations" not in st.session_state or len(st.session_state.durations) != n_notes:
        st.session_state.durations = [DEFAULT_DURATION] * n_notes

    durations: list[float] = st.session_state.durations

    # Build chord track
    chord_data = build_chord_track(notes, durations) if harmonize else None

    # ── 4. Music Sheet & Editor ──
    st.subheader("4 · Music Sheet")
    st.caption("Select a note to change its duration.")

    sel_col1, sel_col2 = st.columns([3, 2])
    with sel_col1:
        selected_idx = st.slider(
            "Note selector",
            0, max(0, n_notes - 1), 0,
            format="Note %d",
            help="Slide to highlight a note on the sheet. Then change its duration below.",
        )
    with sel_col2:
        note_name = _midi_to_note_name(notes[selected_idx]) if notes else ""
        current_dur = durations[selected_idx]
        cur_label = DURATION_LABELS[DURATION_STEPS.index(current_dur)] if current_dur in DURATION_STEPS else "8th"
        st.markdown(f"**Selected:** Note {selected_idx} — **{note_name}** ({cur_label})")

        new_dur_label = st.radio(
            "Duration",
            DURATION_LABELS,
            index=DURATION_STEPS.index(current_dur) if current_dur in DURATION_STEPS else 1,
            horizontal=True,
            key=f"dur_{selected_idx}",
        )
        new_dur = DURATION_STEPS[DURATION_LABELS.index(new_dur_label)]
        if new_dur != current_dur:
            st.session_state.durations[selected_idx] = new_dur
            durations = st.session_state.durations
            # Rebuild chord track with updated durations
            chord_data = build_chord_track(notes, durations) if harmonize else None

    if st.button("Reset all durations to 8th notes"):
        st.session_state.durations = [DEFAULT_DURATION] * n_notes
        durations = st.session_state.durations
        chord_data = build_chord_track(notes, durations) if harmonize else None
        st.rerun()

    # Render VexFlow score
    notation_html = render_notation_html(notes, durations, chord_data, selected_idx)
    st.components.v1.html(notation_html, height=max(200, len(notes) // 12 * 160 + 80), scrolling=True)

    # ── 5. Listen & Download ──
    st.subheader("5 · Listen & Download")

    wav_bytes = synthesize_wav(notes, durations, tempo, chord_data)
    st.audio(wav_bytes, format="audio/wav")

    midi_bytes = generate_midi(notes, durations, tempo, chord_data)
    st.download_button(
        label="Download MIDI",
        data=midi_bytes,
        file_name="sonification.mid",
        mime="audio/midi",
    )

    with st.expander("MIDI note sequence"):
        st.code(" ".join(str(n) for n in notes), language=None)

    if chord_data:
        with st.expander("Chord voicings"):
            lines = []
            for beat, voicing, name, cdur in chord_data:
                v_str = ", ".join(str(n) for n in voicing)
                lines.append(f"Beat {beat:5.1f}  {name:<7s}  [{v_str}]  dur={cdur:.2f}")
            st.code("\n".join(lines), language=None)


if __name__ == "__main__":
    main()
