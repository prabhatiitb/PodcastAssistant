import chromadb
from src.config import client, MODEL_NAME
from src.utils import format_timestamp
from src.chunk_and_index import query_chunks


def answer_question(collection, question: str, top_k: int = 4) -> dict:
    """
    Retrieves relevant chunks and generates a grounded answer with timestamp sources.
    """
    retrieved_chunks = query_chunks(collection, question, top_k=top_k)

    context_block = "\n".join(
        f"[{format_timestamp(c['start'])}] {c['text']}" for c in retrieved_chunks
    )

    prompt = f"""You are answering a student's question about a lecture, using only the transcript excerpts below.
If the excerpts don't contain the answer, say so honestly — do not make anything up.

Transcript excerpts:
{context_block}

Question: {question}

Answer clearly and concisely, based only on the excerpts above.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": retrieved_chunks  # each has "start" and "end" for timestamp links
    }


if __name__ == "__main__":
    client_db = chromadb.PersistentClient(path="data/chroma_db")
    collection = client_db.get_collection(name="episode_<your_episode_id>")  # update this

    test_question = "What is the main topic discussed?"
    result = answer_question(collection, test_question)

    print("\n--- Answer ---\n")
    print(result["answer"])

    print("\n--- Sources ---")
    for src in result["sources"]:
        print(f"  [{format_timestamp(src['start'])}–{format_timestamp(src['end'])}] {src['text'][:80]}...")
