import json
import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def merge_segments_into_chunks(segments: list[dict], max_chunk_words: int = 200) -> list[dict]:
    """
    Merges small Whisper segments into larger, timestamp-aware chunks.
    Each chunk: {"text": str, "start": float, "end": float}
    """
    chunks = []
    current_text = []
    current_word_count = 0
    chunk_start = None

    for seg in segments:
        if chunk_start is None:
            chunk_start = seg["start"]

        current_text.append(seg["text"])
        current_word_count += len(seg["text"].split())

        if current_word_count >= max_chunk_words:
            chunks.append({
                "text": " ".join(current_text),
                "start": chunk_start,
                "end": seg["end"]
            })
            current_text = []
            current_word_count = 0
            chunk_start = None

    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start": chunk_start,
            "end": segments[-1]["end"]
        })

    return chunks


def index_chunks(chunks: list[dict], episode_id: str, persist_dir: str = "data/chroma_db"):
    """
    Embeds chunks and stores them in a ChromaDB collection scoped to this episode.
    """
    db_client = chromadb.PersistentClient(path=persist_dir)
    collection = db_client.get_or_create_collection(name=f"episode_{episode_id}")

    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts).tolist()

    ids = [f"{episode_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"start": c["start"], "end": c["end"]} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return collection


def query_chunks(collection, query: str, top_k: int = 4) -> list[dict]:
    """
    Retrieves the top_k most relevant chunks for a given query.
    """
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": doc,
            "start": meta["start"],
            "end": meta["end"]
        })
    return retrieved


if __name__ == "__main__":
    target_json = "data/transcripts/<your_transcript>.json"  # update this
    episode_id = "<your_episode_id>"  # update this, e.g. the youtube video id

    print("Loading specific transcript...")
    with open(target_json, "r") as f:
        segments = json.load(f)

    # 1. Merge Whisper segments into larger chunks
    chunks = merge_segments_into_chunks(segments)
    print(f"Merged {len(segments)} segments into {len(chunks)} chunks")

    # 2. Index chunks into ChromaDB
    print("Embedding and storing chunks...")
    collection = index_chunks(chunks, episode_id=episode_id)
    print("Successfully indexed chunks")

    # 3. Test retrieval / sanity check
    test_query = "What is the main topic discussed?"
    print(f"\nRunning test query: {test_query}")
    top_results = query_chunks(collection, test_query)

    print("\n--- Retrieved Chunks (Sanity Check) ---")
    for i, res in enumerate(top_results):
        print(f"\nResult {i} [{res['start']}-{res['end']}]: {res['text'][:100]}...")
