# Data Sonification Web App

Converts numeric data into a MIDI melody with optional jazz-chord harmonization, in-browser playback, music sheet notation, and per-note duration editing.

## Technologies
- Python 3.x
- Streamlit
- Pandas
- midiutil
- NumPy
- VexFlow (JS)

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run sonification_app.py
   ```

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click "New app".
4. Select your repository, branch, and main file (`sonification_app.py`).
5. Click "Deploy".
