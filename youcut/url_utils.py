import re


def normalize_video_url(url: str) -> str:
    """Normalize URLs copied from shells/docs with backslash-escaped separators."""
    return re.sub(r"\\([?=&])", r"\1", url.strip())
