from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

from youcut.models import VideoMetadata
from youcut.video_metadata import VideoMetadataError, fetch_metadata
from youcut.yt_dlp_auth import YtDlpAuthConfig

VALID_INFO = {
    "title": "My Live Stream",
    "duration": 3600,
    "webpage_url": "https://youtube.com/watch?v=abc123",
}

URL = "https://youtube.com/watch?v=abc123"


def _make_ydl_mock(info):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = info
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx, mock_ydl


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_returns_video_metadata(mock_ydl_cls):
    mock_ctx, mock_ydl = _make_ydl_mock(VALID_INFO)
    mock_ydl_cls.return_value = mock_ctx

    result = fetch_metadata(URL)

    assert isinstance(result, VideoMetadata)
    assert result.title == "My Live Stream"
    assert result.duration_seconds == 3600.0
    assert result.url == URL


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_passes_download_false(mock_ydl_cls):
    mock_ctx, mock_ydl = _make_ydl_mock(VALID_INFO)
    mock_ydl_cls.return_value = mock_ctx

    fetch_metadata(URL)

    mock_ydl.extract_info.assert_called_once_with(URL, download=False)


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_raises_on_none_info(mock_ydl_cls):
    mock_ctx, mock_ydl = _make_ydl_mock(None)
    mock_ydl_cls.return_value = mock_ctx

    with pytest.raises(VideoMetadataError, match="No metadata returned"):
        fetch_metadata(URL)


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_raises_on_missing_duration(mock_ydl_cls):
    info = {"title": "My Live", "webpage_url": URL}
    mock_ctx, mock_ydl = _make_ydl_mock(info)
    mock_ydl_cls.return_value = mock_ctx

    with pytest.raises(VideoMetadataError, match="missing duration"):
        fetch_metadata(URL)


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_raises_on_download_error(mock_ydl_cls):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("Video unavailable")
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ydl_cls.return_value = mock_ctx

    with pytest.raises(VideoMetadataError, match="Could not access video"):
        fetch_metadata(URL)


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_raises_on_network_error(mock_ydl_cls):
    mock_ctx, mock_ydl = _make_ydl_mock(None)
    mock_ydl.extract_info.side_effect = OSError("Network unreachable")
    mock_ydl_cls.return_value = mock_ctx

    with pytest.raises(VideoMetadataError, match="Could not access video"):
        fetch_metadata(URL)


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_duration_as_float(mock_ydl_cls):
    info = {**VALID_INFO, "duration": 90.5}
    mock_ctx, mock_ydl = _make_ydl_mock(info)
    mock_ydl_cls.return_value = mock_ctx

    result = fetch_metadata(URL)

    assert result.duration_seconds == 90.5


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_normalizes_shell_escaped_url(mock_ydl_cls):
    mock_ctx, mock_ydl = _make_ydl_mock(VALID_INFO)
    mock_ydl_cls.return_value = mock_ctx

    result = fetch_metadata(r"https://www.youtube.com/watch\?v\=abc123")

    mock_ydl.extract_info.assert_called_once_with("https://www.youtube.com/watch?v=abc123", download=False)
    assert result.url == "https://www.youtube.com/watch?v=abc123"


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_passes_cookies_from_browser(mock_ydl_cls):
    mock_ctx, _ = _make_ydl_mock(VALID_INFO)
    mock_ydl_cls.return_value = mock_ctx

    fetch_metadata(URL, auth_config=YtDlpAuthConfig(browser="chrome"))

    call_opts = mock_ydl_cls.call_args[0][0]
    assert call_opts["cookiesfrombrowser"] == ("chrome",)


@patch("youcut.video_metadata.yt_dlp.YoutubeDL")
def test_fetch_metadata_appends_cookie_hint_on_bot_check(mock_ydl_cls):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ydl_cls.return_value = mock_ctx

    with pytest.raises(VideoMetadataError, match="YOUCUT_COOKIES_FROM_BROWSER=chrome"):
        fetch_metadata(URL)
