# humanitexts

Tools for transcribing and editing lecture and long-form audio into cleaner written prose.

---

## Overview

This repository contains a set of scripts developed for working with recorded lectures and podcast-style audio in the humanities. The aim is not simply transcription, but the production of readable, lightly edited text that preserves the content, tone, and structure of spoken lectures while making them usable for study, teaching, and archival purposes.

The workflow reflects a practical need: converting large volumes of recorded material—university lectures, discussions, and long-form audio—into coherent written form without reducing or summarizing the content.

---

## Approach

The pipeline combines automated transcription with constrained language-model editing:

1. **Transcription (Whisper)**

   * Audio is transcribed using OpenAI’s Whisper models.
   * Settings are tuned for lecture-style audio, including speech over music and variable clarity.

2. **Segmentation**

   * Transcripts are divided into sentence-based chunks (~500 words).
   * This allows for controlled processing and avoids degradation over long inputs.

3. **Editing (GPT models)**

   * Chunks are edited for clarity, grammar, and readability.
   * The process is intentionally conservative:

     * no summarization
     * no removal of substantive content
     * preservation of voice and structure

4. **Recombination**

   * Edited chunks are reassembled into a complete document.
   * Output is typically in Markdown format for further use.

---

## Quick Start

The simplest way to use the system is:

```bash
# set up environment (recommended)
python -m venv .venv
source .venv/bin/activate

pip install openai spacy
python -m spacy download en_core_web_sm

# run the pipeline
python lecture_pipeline_whisper_mini_markdown.py your_audio_file.mp3
```

This will:

* transcribe the audio using Whisper
* split the transcript into manageable chunks
* apply a light editing pass
* produce a `.md` file in the same directory

---

### Batch processing

To process an entire folder:

```bash
python lecture_pipeline_whisper_mini_markdown.py .
```

---

### Notes

* Output files are written alongside the source audio
* Intermediate files are stored in `_transcription_work/`
* If a transcript already exists, Whisper will be skipped

---

### Execution Notes

These scripts are designed to be run inside a Python virtual environment.

When a virtual environment is activated, both `python` and `python3` should resolve to the same interpreter. This ensures that dependencies such as `spacy` are available to the scripts.

If you encounter errors like:

```
ModuleNotFoundError: No module named 'spacy'
```

it usually indicates that the script is being executed with a different Python interpreter than the one in which dependencies were installed.

Running the scripts via:

```bash
python script_name.py
```

inside an activated environment is recommended.

---

## Scripts

The repository currently includes several primary workflows:

### `lecture_pipeline_whisper_4_and_5.py`

Full lecture processing pipeline.

* Input: audio files
* Output: two versions of the transcript

  * GPT-5 (more assertive editing)
  * GPT-4.1-mini (lighter editing)

Use this for high-quality lecture transcription and comparison of editing styles.

---

### `lecture_pipeline_whisper_mini_markdown.py`

High-throughput pipeline for batch processing.

* Input: audio files
* Output: Markdown transcripts

Uses a lighter model for speed and cost efficiency. Suitable for large collections (e.g. podcast archives).

---

### `lecture_pipeline_whisper.mini.py`

Streamlined high-throughput pipeline.

* Input: audio files
* Output: Markdown transcripts

This version emphasizes simplicity and speed:

* uses Whisper (typically medium model)
* applies a single GPT-4.1-mini editing pass
* produces clean Markdown output with minimal overhead

Designed for:

* large-scale batch processing
* lower-cost transcription workflows
* situations where a single-pass edit is sufficient

Compared to other pipelines, this script reduces complexity while maintaining a readable output.

---

### `lecture_pipeline_edit_existing_markdown.py`

Second-pass refinement.

* Input: existing Markdown transcripts
* Output: lightly improved versions

Applies a constrained editing pass to improve readability without altering structure or meaning.

This script is useful for reprocessing transcripts after transcription or for applying improvements to existing Markdown files.

---

## Requirements

Install the required Python packages:

```bash
pip install openai spacy
python -m spacy download en_core_web_sm
```

You will also need:

* Whisper CLI installed and available in your PATH
* an OpenAI API key set as an environment variable

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## Design Principles

* **Preservation over transformation**
  The goal is not to rewrite lectures, but to make them readable.

* **Humanistic use case**
  The scripts are designed for teaching, research, and archival work in the humanities.

* **Scalability**
  The system is built to handle large collections of audio with minimal intervention.

* **Transparency**
  Each stage of the process is explicit and inspectable.

---

## Status

This is an active working repository. The scripts reflect an evolving workflow rather than a finalized software package.

Documentation will be expanded over time as the system is refined.

---

## Notes

The project grows out of practical teaching needs rather than software engineering priorities. It is shared here in the hope that others working with similar materials—lectures, seminars, long-form audio—may find it useful or adapt
