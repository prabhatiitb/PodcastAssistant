import os
import yt_dlp


def get_video_duration(url: str) -> int:
    """Returns video duration in seconds without downloading it."""
    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("duration", 0)


def download_audio_from_youtube(url: str, output_dir: str = "data/audio") -> str:
    """
    Downloads audio from a YouTube URL and converts it to mp3.
    Returns the path to the downloaded audio file.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            filepath = os.path.join(output_dir, f"{video_id}.mp3")
            return filepath
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Could not download audio: {e}")


def handle_uploaded_file(uploaded_file, output_dir: str = "data/audio") -> str:
    """
    Saves an uploaded MP3 (from Streamlit's file_uploader) to disk.
    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath


if __name__ == "__main__":
    test_url = "<paste a youtube url here>"
    duration = get_video_duration(test_url)
    print(f"Video duration: {duration} seconds")

    path = download_audio_from_youtube(test_url)
    print(f"Downloaded to: {path}")
