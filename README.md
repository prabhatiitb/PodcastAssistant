# 🎧 Podcast Assistant

A RAG-based tool that helps students decide whether a long lecture/podcast is worth their time — and if it is, lets them ask questions and jump straight to the relevant moment.

## Problem

Students juggling multiple courses accumulate long lecture recordings and podcasts (45–90 min) with no fast way to judge whether an episode is worth listening to, or to find the specific segment relevant to what they're studying. Unlike text, audio can't be skimmed — you either commit to listening or skip it blind.

## Solution

A two-tier tool:
1. **Skim layer** — on upload, auto-generates a chronological topic outline with timestamps, so a student can decide "listen or skip" in under 30 seconds.
2. **Drill-down layer** — a Q&A chat interface where the student asks specific questions and gets answers grounded in the transcript, with clickable timestamp links to the exact moment.

This differs from typical "chat with your podcast" RAG demos, which usually stop at Q&A — the outline/skim layer solves the upstream decision problem (should I even listen to this?), not just the downstream retrieval problem.

## Architecture

```
YouTube URL / MP3 upload
        │
        ▼
  yt-dlp (audio extraction)
        │
        ▼
  faster-whisper, local, "small" model (transcription with timestamps)
        │
        ▼
  Chunking (merge Whisper segments into ~200-word, timestamp-aware chunks)
        │
        ▼
  sentence-transformers embeddings → ChromaDB (per-episode collection)
        │
        ├──► Tier 1: Full transcript → Gemini → topic outline
        │
        └──► Tier 2: Query → embed → retrieve top-k chunks → Gemini → grounded answer + timestamps
                │
                ▼
          Streamlit UI
```

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Audio extraction | yt-dlp | Handles YouTube's muxed audio/video streams |
| Transcription | faster-whisper (local, `small`, int8) | Free, no API cost, runs on CPU (M1 Mac) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, local, fast enough for this scale |
| Vector store | ChromaDB | Simple, local-first, per-episode collections |
| LLM | Gemini (`gemini-3.6-flash`) | Free tier, large context window for full-transcript summarization |
| UI | Streamlit | Fast to build, good fit for a chat-style interface |

## Key Design Decisions

- **Word-count-based chunking, not fixed time windows** — speech rate varies, so chunking by ~200 words gives more consistent semantic density per chunk than a fixed time window would.
- **Single-episode scope** — each episode gets its own ChromaDB collection. Keeps retrieval automatically scoped with no cross-episode contamination risk, at the cost of not supporting cross-episode search (a reasonable scope cut for an MVP).
- **Explicit hallucination guardrail** — the Q&A prompt instructs the model to say "not covered in this transcript" rather than guessing, and this is directly tested in the evaluation set.
- **Full transcript for summarization, retrieved chunks for Q&A** — summarization needs global context (can't summarize from 4 chunks), so it intentionally does *not* use the same retrieval path as Q&A.

## Evaluation

Built a small hand-labeled test set (`evals/test_questions.json`) checking:
- **Retrieval hit rate** — does the system retrieve a chunk from the correct timestamp range for a given question?
- **Answer accuracy** — simple substring match against an expected answer fragment.
- **Hallucination guardrail** — includes a deliberately unanswerable question to confirm the system says so rather than fabricating an answer.

This uses lightweight substring matching, which is a reasonable check at this scale but has real limits — a production system would use LLM-as-judge scoring for more nuanced correctness evaluation.

Run it with:
```bash
python3 -m evals.run_eval
```

## Setup & Running Locally

```bash
git clone <your-repo-url>
cd PodcastAssistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your GEMINI_API_KEY to .env

streamlit run app.py
```

## Deployment Notes

Deployed on Streamlit Community Cloud with a duration cap (~15 min) on input audio to stay within free-tier compute limits, since local Whisper transcription is resource-intensive in a shared cloud environment. The architecture supports longer content given more compute.

## Possible Extensions

- Cross-episode search (would require episode-ID metadata filtering instead of per-episode collections)
- LLM-as-judge evaluation instead of substring matching
- Seekable audio player for uploaded MP3s (currently only YouTube gets clickable timestamp links)

## Live Demo

[Add your deployed Streamlit URL here]
