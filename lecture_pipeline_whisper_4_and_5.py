#!/usr/bin/env python3


# PURPOSE:
# Full lecture processing pipeline (primary production script)

# INPUT:
# - Audio files (.mp3, .mp4, .m4a, .wav)

# OUTPUT:
# - .cleaned_5.md (GPT-5 version)
# - .cleaned_mini.md (GPT-4.1-mini version)

# DESCRIPTION:
# - Runs Whisper transcription (large-v3)
# - Splits transcript into chunks (~500 words)
# - Processes chunks with both GPT-5 and GPT-4.1-mini
# - Recombines into two full lecture outputs

# USE CASE:
# - Standard lecture processing
# - High-quality, dual-model comparison output

# NOTES:
# - Primary “go-to” script for course lectures

import sys
from pathlib import Path

import spacy
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

WHISPER_MODEL = "large-v3"

OPENAI_MODEL_5 = "gpt-5"
OPENAI_MODEL_MINI = "gpt-4.1-mini"

CHUNK_WORDS = 500

WORK_DIR = "_transcription_work"

# ==============================
# PROMPTS
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
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'"
    }

    for k,v in replacements.items():
        text = text.replace(k,v)

    return text

# ==============================
# WHISPER (CLI via python -m whisper)
# ==============================

def transcribe(audio, work_dir):

    txt = work_dir / f"{audio.stem}.txt"

    print("Running Whisper CLI...")

    subprocess.run([
        sys.executable, "-m", "whisper",
        str(audio),
        "--model", WHISPER_MODEL,
        "--output_format", "txt",
        "--output_dir", str(work_dir)
    ], check=True)

    return txt if txt.exists() else None

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
# OPENAI EDIT (STABLE)
# ==============================

def edit_file(file, work_dir, model, prompt, suffix):

    client = OpenAI()

    text = file.read_text()

    full_prompt = prompt + "\n\n" + text

    r = client.responses.create(
        model=model,
        input=full_prompt
    )

    cleaned = enforce_ascii(r.output_text)

    out = work_dir / f"{file.stem}-{suffix}.txt"
    out.write_text(cleaned)

    return out

def edit_all(files, work_dir, model, prompt, suffix):

    edited=[]

    print(f"Total chunks ({suffix}): {len(files)}")

    for f in files:
        try:
            result = edit_file(f, work_dir, model, prompt, suffix)
            print(f"✓ Edited ({suffix}): {f.name}")
            edited.append(result)
        except Exception as e:
            print(f"✗ FAILED ({suffix}): {f.name} -> {e}")

    return edited

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

    final_5 = audio.with_suffix(".cleaned_5.md")
    final_mini = audio.with_suffix(".cleaned_mini.md")

    if final_5.exists() and final_mini.exists():
        print("Both outputs exist - skipping:", audio.name)
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

    print("Editing chunks (GPT-5)")
    edited_5 = edit_all(splits, work_dir, OPENAI_MODEL_5, SYSTEM_PROMPT, "5")

    print("Editing chunks (4.1-mini)")
    edited_mini = edit_all(splits, work_dir, OPENAI_MODEL_MINI, SYSTEM_PROMPT_MINI, "mini")

    combine(edited_5, final_5)
    combine(edited_mini, final_mini)

    print("Finished:", final_5)
    print("Finished:", final_mini)

# ==============================
# MAIN
# ==============================

def main():

    if len(sys.argv)!=2:
        sys.exit("Usage: lecture_pipeline_4_and_5.py <audio_or_folder>")

    p = Path(sys.argv[1])

    if p.is_file():
        process(p)
    else:
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in [".mp3", ".mp4", ".m4a", ".wav"]:
                process(f)

if __name__ == "__main__":
    main()
