#!/usr/bin/env python3

# PURPOSE:
# High-throughput transcription and cleaning pipeline (fast batch processing)

# INPUT:
# - Audio files (.mp3, .mp4, .m4a, .wav)

# OUTPUT:
# - .md (cleaned markdown transcript)

# DESCRIPTION:
# - Runs Whisper transcription (typically medium model)
# - Splits transcript into chunks
# - Processes chunks with GPT-4.1-mini only
# - Recombines into a single markdown output

# USE CASE:
# - Large batch processing (e.g., In Our Time archive)
# - Fast, lower-cost transcript generation

# NOTES:
# - Optimized for speed and scale
# - Includes thread limits for Whisper CPU control

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

import os
import spacy
from openai import OpenAI

# ==============================
# THREAD CONTROL (NEW)
# ==============================

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# ==============================
# CONFIG
# ==============================

WHISPER_MODEL = "medium"

OPENAI_MODEL_MINI = "gpt-4.1-mini"

CHUNK_WORDS = 500
MAX_WORKERS = 4

WORK_DIR = "_transcription_work"

client = OpenAI()

# ==============================
# PROMPT
# ==============================

SYSTEM_PROMPT_MINI = """
Edit the following text for clarity, readability, and flow.

You may:
- remove verbal fillers and repetition typical of speech
- smooth phrasing and sentence structure
- improve grammar and punctuation

Do NOT:
- change meaning
- introduce new ideas
- remove important content
- alter names, titles, or technical terms

Preserve the author’s voice, but make the text read more like clear written prose rather than speech.

Return the edited text only.
"""

# ==============================
# SPACY
# ==============================

nlp = spacy.load("en_core_web_sm", disable=["parser","tagger","ner"])
nlp.add_pipe("sentencizer")

# ==============================
# ASCII CLEANER
# ==============================

def enforce_ascii(text):
    replacements = {
        "":"-",
        "":"-",
        "“":'"',
        "”":'"',
        "’":"'",
        "‘":"'"
    }
    for k,v in replacements.items():
        text = text.replace(k,v)
    return text

# ==============================
# WHISPER
# ==============================

def transcribe(audio, work_dir):

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"

    subprocess.run([
        "whisper",
        str(audio),
        "--model", WHISPER_MODEL,
        "--language", "en",
        "--condition_on_previous_text", "False",
        "--no_speech_threshold", "0.1",
        "--output_format", "txt",
        "--output_dir", str(work_dir)
    ], check=True, env=env)

    txt = work_dir / f"{audio.stem}.txt"

    if txt.exists():
        return txt

    print("WARNING: Whisper output not found:", audio)
    return None

# ==============================
# NORMALIZE
# ==============================

def normalize(txt):
    text = txt.read_text()
    text = text.replace("\n"," ")
    text = " ".join(text.split())
    txt.write_text(text)

# ==============================
# SPLIT
# ==============================

def split(txt, work_dir):

    text = txt.read_text()

    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    chunks = []
    cur = []
    words = 0

    for s in sentences:

        wc = len(s.split())

        if words + wc > CHUNK_WORDS and cur:
            chunks.append(" ".join(cur))
            cur=[]
            words=0

        cur.append(s)
        words+=wc

    if cur:
        chunks.append(" ".join(cur))

    base = txt.stem
    outdir = work_dir / base
    outdir.mkdir(exist_ok=True)

    files=[]

    for i,c in enumerate(chunks,1):
        f = outdir / f"{base}-{i:02d}.txt"
        f.write_text(c)
        files.append(f)

    return files

# ==============================
# MINI EDIT
# ==============================

def edit_file(file, work_dir):

    text = file.read_text()

    r = client.responses.create(
        model=OPENAI_MODEL_MINI,
        input=[
            {"role":"system","content":SYSTEM_PROMPT_MINI},
            {"role":"user","content":text}
        ]
    )

    cleaned = enforce_ascii(r.output_text)

    out = work_dir / f"{file.stem}-mini.txt"
    out.write_text(cleaned)

    return out

def edit_all(files, work_dir):

    edited=[]

    print(f"Total chunks (mini): {len(files)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(edit_file, f, work_dir): f
            for f in files
        }

        for future in as_completed(futures):

            f = futures[future]

            try:
                result = future.result()
                print(f"✓ Edited (mini): {f.name}")
                edited.append(result)

            except Exception as e:
                print(f"✗ FAILED (mini): {f.name} -> {e}")

    return sorted(edited)

# ==============================
# COMBINE
# ==============================

def combine(files,output):

    combined=""

    for f in files:
        combined += f.read_text() + "\n\n"

    combined = enforce_ascii(combined)

    output.write_text(combined)

# ==============================
# PIPELINE
# ==============================

def process(audio):

    print("Processing:", audio)

    work_dir = audio.parent / WORK_DIR
    work_dir.mkdir(exist_ok=True)

    final_md = audio.with_suffix(".md")

    if final_md.exists():
        print("Already processed - skipping:", audio.name)
        return

    work_txt = work_dir / f"{audio.stem}.txt"

    if work_txt.exists():
        print("Transcript exists - skipping Whisper")
    else:
        txt = transcribe(audio, work_dir)
        if txt is None:
            return

    normalize(work_txt)

    print("Splitting transcript")
    splits = split(work_txt, work_dir)

    print("Editing chunks (mini)")
    edited = edit_all(splits, work_dir)

    combine(edited, final_md)

    print("Finished:", final_md)

# ==============================
# MAIN
# ==============================

def main():

    if len(sys.argv)!=2:
        sys.exit("Usage: inourtime_pipeline_audio.py <audio_or_folder>")

    p = Path(sys.argv[1])

    if p.is_file():
        process(p)
    else:
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in [".mp3", ".mp4", ".m4a", ".wav"]:
                process(f)

if __name__ == "__main__":
    main()
