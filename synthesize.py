import soundfile as sf
import sys
import os

from pretty_midi import PrettyMIDI

SR = 44100
SF_PATH = "soundfonts/Standard_Drum_Kit.sf2"

def synthesize_all(source_dir, audio_dir, prefix=""):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory {source_dir} does not exist.")
    
    filesSytnhesized = 0
    for filename in os.listdir(source_dir):
        if filename.endswith(".mid"):
            midi_path = f"{source_dir}/{filename}"
            midi_to_audio(midi_path, audio_dir, prefix=prefix)
            filesSytnhesized += 1
    
    print(f"Synthesized {filesSytnhesized} files to {audio_dir}")

def midi_to_audio(source_path, audio_dir, prefix,sr=SR):
    pm = PrettyMIDI(source_path)
    audio = pm.fluidsynth(fs=sr, sf2_path=SF_PATH)

    sf.write(f'{audio_dir}/{prefix}_{getFilename(source_path)}.wav', audio, sr, 'PCM_24')

def getFilename(midi_path):
    filename = midi_path.split("/")[-1]
    filename = filename.split(".")[0]
    return filename

def usageAndExit():
    print("Usage: python synthesize.py <dir>")
    sys.exit(1)