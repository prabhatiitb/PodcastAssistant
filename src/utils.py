def format_timestamp(seconds: float) -> str:
    """Converts seconds into MM:SS format for display."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def make_timestamp_link(youtube_url: str, seconds: float) -> str:
    """Builds a YouTube URL that jumps to a specific timestamp."""
    base_url = youtube_url.split("&")[0]  # strip existing params
    return f"{base_url}&t={int(seconds)}s"
