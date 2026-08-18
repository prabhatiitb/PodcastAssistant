from src.config import client, MODEL_NAME
from src.utils import format_timestamp


def generate_outline(chunks: list[dict]) -> str:
    """
    Generates a skimmable topic outline with timestamps from ordered chunks.
    """
    transcript_block = "\n".join(
        f"[{format_timestamp(c['start'])}] {c['text']}" for c in chunks
    )

    prompt = f"""You are summarizing a lecture/podcast transcript for a student deciding whether to listen to it.

Transcript (with timestamps):
{transcript_block}

Produce a concise outline of the main topics covered, in chronological order.
For each topic, include the approximate timestamp range where it's discussed.
Format as a bulleted list: "[start–end] Topic description"
Keep it skimmable — a student should be able to read this in under 30 seconds
and decide if the content is relevant to them.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    import json
    from src.chunk_and_index import merge_segments_into_chunks

    transcript_path = "data/transcripts/<your_transcript>.json"  # update this
    with open(transcript_path) as f:
        segments = json.load(f)

    chunks = merge_segments_into_chunks(segments)
    outline = generate_outline(chunks)

    print("\n--- Generated Outline ---\n")
    print(outline)
