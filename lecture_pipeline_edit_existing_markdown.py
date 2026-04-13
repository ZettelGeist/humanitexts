#!/usr/bin/env python3

# PURPOSE:
# Second-pass refinement of existing markdown lecture files

# INPUT:
# - Existing .md files

# OUTPUT:
# - -mini.md (refined version)

# DESCRIPTION:
# - Reads completed markdown transcripts
# - Normalizes and splits into chunks
# - Applies GPT-4.1-mini editing pass
# - Recombines into improved markdown output

# USE CASE:
# - Light smoothing and readability improvement
# - Post-processing after initial transcription

# NOTES:
# - Does NOT run Whisper
# - Safe to run on completed lecture files

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import spacy
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

OPENAI_MODEL_MINI = "gpt-4.1-mini"

CHUNK_WORDS = 500
MAX_WORKERS = 2

WORK_DIR = "_transcription_work"

client = OpenAI()

# ==============================
# PROMPT
# ==============================

SYSTEM_PROMPT_MINI = """
Lightly edit the text for clarity and readability.

You may:
- fix grammar and punctuation
- smooth phrasing slightly
- repair obvious transcription errors if clearly identifiable

Do NOT:
- remove meaningful repetition
- alter tone or certainty
- simplify complex ideas
- remove qualifications or nuance

Preserve meaning and conversational structure.

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
        "—":"-",
        "–":"-",
        "“":'"',
        "”":'"',
        "’":"'",
        "‘":"'"
    }
    for k,v in replacements.items():
        text = text.replace(k,v)
    return text

# ==============================
# NORMALIZE
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
            cur = []
            words = 0

        cur.append(s)
        words += wc

    if cur:
        chunks.append(" ".join(cur))

    base = txt.stem
    outdir = work_dir / base
    outdir.mkdir(exist_ok=True)

    files = []

    for i, c in enumerate(chunks, 1):
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

    edited = []

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

def combine(files, output):

    combined = ""

    for f in files:
        combined += f.read_text() + "\n\n"

    combined = enforce_ascii(combined)

    output.write_text(combined)

# ==============================
# PROCESS MD FILE
# ==============================

def process_md(md_file):

    final_mini = md_file.with_name(md_file.stem + "-mini.md")

    print("Processing:", md_file)

    work_dir = md_file.parent / WORK_DIR
    work_dir.mkdir(exist_ok=True)

    work_txt = work_dir / f"{md_file.stem}.txt"

    text = md_file.read_text()
    work_txt.write_text(text)

    normalize(work_txt)

    print("Splitting transcript")
    splits = split(work_txt, work_dir)

    print("Editing chunks (mini)")
    edited_mini = edit_all(splits, work_dir)

    combine(edited_mini, final_mini)

    print("Finished:", final_mini)

# ==============================
# MAIN
# ==============================

def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: lecture_pipeline_markdown.py <md_file_or_folder>")

    p = Path(sys.argv[1])

    if p.is_file():
        process_md(p)

    else:
        for f in sorted(p.glob("*.md")):

            name = f.name

            # Only process ORIGINAL files
            if "-mini" in name or "-5" in name:
                continue

            # Skip if already processed
            mini_version = f.with_name(f.stem + "-mini.md")
            if mini_version.exists():
                print("Skipping (already processed):", f.name)
                continue

            process_md(f)

if __name__ == "__main__":
    main()
