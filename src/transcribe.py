import json
import os
from faster_whisper import WhisperModel


def transcribe_audio(audio_path: str, model_size: str = "small") -> list[dict]:
    """
    Transcribes an audio file and returns a list of timestamped segments.
    Each segment: {"start": float, "end": float, "text": str}
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, beam_size=5)

    result = []
    for segment in segments:
        result.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
    return result


def save_transcript(segments: list[dict], audio_filename: str, output_dir: str = "data/transcripts") -> str:
    """
    Saves the segmented transcript as JSON for downstream chunking/indexing.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)

    return output_path


if __name__ == "__main__":
    test_audio_path = "data/audio/<your_audio_file>.mp3"  # update this

    segments = transcribe_audio(test_audio_path)
    output_path = save_transcript(segments, test_audio_path)

    print(f"\nTranscript saved to: {output_path}")
    print(f"Total segments: {len(segments)}")
