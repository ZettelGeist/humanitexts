#!/usr/bin/env python3

# PURPOSE:
# Single-model lecture processing pipeline (testing and controlled runs)

# INPUT:
# - Audio files (.mp3, .mp4, .m4a, .wav)

# OUTPUT:
# - .cleaned.md (single model output)

# DESCRIPTION:
# - Runs Whisper transcription
# - Splits transcript into chunks
# - Processes chunks using a selected OpenAI model (GPT-5, 4.1, or mini)
# - Recombines into one cleaned lecture file

# USE CASE:
# - Model comparison and testing
# - Faster or simplified lecture runs

# NOTES:
# - Model selection controlled in CONFIG section
# - Includes thread limits for Whisper CPU control

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import spacy
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

WHISPER_MODEL = "large-v3"
OPENAI_MODEL = "gpt-5"
# OPENAI_MODEL = "gpt-4.1"
# OPENAI_MODEL = "gpt-4.1-mini"

CHUNK_WORDS = 500
MAX_WORKERS = 4

WORK_DIR = "_transcription_work"

client = OpenAI()

# ==============================
# PROMPT
# ==============================

SYSTEM_PROMPT = """
You are editing a transcript of a university lecture intended for undergraduate teaching in European cultural and intellectual history. These transcripts are part of a broader project to convert recorded lecture material into polished, college-level instructional prose for a textbook or lecture companion. Lightly edit the text for grammar, punctuation, and flow.

Rules:
- Do NOT summarize, condense, paraphrase, or reorganize.
- Do NOT remove repetition or emphasis.
- Preserve all substantive points made in the lectures
- Do not cut or summarize content.
- Transform oral phrasing into clear, structured written prose
- Maintain a college-level introductory textbook tone.
- Preserve voice and cadence.
- Make only minimal corrections required for readability.
- Words to minimize or avoid: "akin", "delve"
- All output must use ASCII-safe characters only
- Use hyphens - instead of em dashes
- Standard quotation marks "..."

If Whisper has transcribed a foreign name or term phonetically
(German, French, Italian, Latin), correct it if the intended word
can be confidently identified from context.

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

    subprocess.run([
        "whisper",
        str(audio),
        "--model", WHISPER_MODEL,
        "--task", "transcribe",
        "--output_dir", str(work_dir)
        ], check=True)

    txt = work_dir / f"{audio.stem}.txt"

    if txt.exists():
        return txt

    for ext in [".vtt", ".srt"]:
        alt = work_dir / f"{audio.stem}{ext}"
        if alt.exists():
            print(f"Converting {alt.name} to txt")

            lines = []
            for line in alt.read_text().splitlines():
                if "-->" in line:
                    continue
                if line.strip().isdigit():
                    continue
                lines.append(line)

            txt.write_text(" ".join(lines))
            return txt

    print("WARNING: Whisper output not found:", audio)
    return None

# ==============================
# CLEAN TRANSCRIPT
# ==============================

def normalize(txt):

    text = txt.read_text()

    text = text.replace("\n"," ")
    text = " ".join(text.split())

    txt.write_text(text)

# ==============================
# SPLIT INTO CHUNKS
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
# OPENAI EDIT
# ==============================

def edit_file(file, work_dir):

    text = file.read_text()

    r = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":text}
        ]
    )
    
    cleaned = enforce_ascii(r.output_text)

    out = work_dir / f"{file.stem}-edited.txt"
    out.write_text(cleaned)

    return out

def edit_all(files, work_dir):

    edited=[]
    failed=[]

    print(f"Total chunks: {len(files)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {executor.submit(edit_file,f,work_dir): f for f in files}

        for future in as_completed(futures):

            f = futures[future]

            try:
                result = future.result()
                print(f"✓ Edited: {f.name}")
                edited.append(result)

            except Exception as e:
                print(f"✗ FAILED: {f.name} -> {e}")
                failed.append(f)

    if failed:
        print("\nFAILED CHUNKS:")
        for f in failed:
            print(f"- {f.name}")

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

    final = audio.with_suffix(".cleaned.md")

    if final.exists():
        print("Final already exists - skipping:", final.name)
        return

    work_txt = work_dir / f"{audio.stem}.txt"

    if work_txt.exists():
        print("Transcript already exists - skipping Whisper")
    else:
        txt = transcribe(audio, work_dir)

        if txt is None:
            print("Skipping due to missing transcript:", audio)
            return

    normalize(work_txt)

    print("Splitting transcript")

    splits = split(work_txt, work_dir)

    print("Editing chunks")

    edited = edit_all(splits, work_dir)

    combine(edited, final)

    print("Finished:", final)

# ==============================
# MAIN
# ==============================

def main():

    if len(sys.argv)!=2:
        sys.exit("Usage: lecture_pipeline.py <audio_or_folder>")

    p = Path(sys.argv[1])

    if p.is_file():
        process(p)
    else:
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in [".mp3", ".mp4", ".m4a", ".wav"]:
                process(f)

if __name__ == "__main__":
    main()
