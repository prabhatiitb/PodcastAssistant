import streamlit as st
import os

from src.extract_audio import download_audio_from_youtube, handle_uploaded_file
from src.transcribe import transcribe_audio, save_transcript
from src.chunk_and_index import merge_segments_into_chunks, index_chunks
from src.summarize import generate_outline
from src.qa import answer_question
from src.utils import format_timestamp, make_timestamp_link


st.set_page_config(page_title="Podcast Assistant", layout="centered")
st.title("🎧 Podcast Assistant")
st.caption("Paste a YouTube link or upload an MP3 — get a skim-friendly outline, then ask questions with timestamped answers.")

# ---- Session state init ----
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.collection = None
    st.session_state.outline = None
    st.session_state.chat_history = []
    st.session_state.youtube_url = None
    st.session_state.audio_path = None


# ---- Input section ----
input_mode = st.radio("Choose input type:", ["YouTube URL", "Upload MP3"])

youtube_url = None
uploaded_file = None

if input_mode == "YouTube URL":
    youtube_url = st.text_input("Paste a YouTube URL")
else:
    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

process_btn = st.button("Process Episode", type="primary")


# ---- Processing pipeline ----
if process_btn:
    if input_mode == "YouTube URL" and not youtube_url:
        st.error("Please paste a YouTube URL.")
    elif input_mode == "Upload MP3" and not uploaded_file:
        st.error("Please upload an MP3 file.")
    else:
        with st.spinner("Step 1/4: Extracting audio..."):
            try:
                if input_mode == "YouTube URL":
                    audio_path = download_audio_from_youtube(youtube_url)
                    st.session_state.youtube_url = youtube_url
                else:
                    audio_path = handle_uploaded_file(uploaded_file)
                    st.session_state.youtube_url = None
                st.session_state.audio_path = audio_path
            except Exception as e:
                st.error(f"Audio extraction failed: {e}")
                st.stop()

        with st.spinner("Step 2/4: Transcribing audio (this can take a minute)..."):
            segments = transcribe_audio(audio_path)
            episode_id = os.path.splitext(os.path.basename(audio_path))[0]
            save_transcript(segments, audio_path)

        with st.spinner("Step 3/4: Chunking and indexing..."):
            chunks = merge_segments_into_chunks(segments)
            collection = index_chunks(chunks, episode_id=episode_id)
            st.session_state.collection = collection

        with st.spinner("Step 4/4: Generating outline..."):
            outline = generate_outline(chunks)
            st.session_state.outline = outline

        st.session_state.processed = True
        st.session_state.chat_history = []  # reset chat for new episode
        st.success("Done! See the outline and ask questions below.")


# ---- Outline display (Tier 1) ----
if st.session_state.processed:
    st.subheader("📋 Episode Outline")
    st.markdown(st.session_state.outline)

    st.divider()

    # ---- Q&A chat (Tier 2) ----
    st.subheader("💬 Ask about this episode")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about this episode...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(st.session_state.collection, question)
                answer_text = result["answer"]

                st.markdown(answer_text)

                st.caption("Sources:")
                for src in result["sources"]:
                    ts_label = f"[{format_timestamp(src['start'])}–{format_timestamp(src['end'])}]"
                    if st.session_state.youtube_url:
                        link = make_timestamp_link(st.session_state.youtube_url, src["start"])
                        st.markdown(f"- [{ts_label}]({link}) {src['text'][:100]}...")
                    else:
                        st.markdown(f"- {ts_label} {src['text'][:100]}...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
