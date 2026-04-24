"""Unit tests for YouTubeUploader."""
from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from youcut.uploader.base import ClipMetadata, UploadResult
from youcut.uploader.youtube import YouTubeUploader, _credentials_to_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(**kwargs) -> ClipMetadata:
    defaults = dict(
        title="Test Title",
        description="A great description",
        hashtags=["#test", "#viral"],
        caption="A great description\n\n#test #viral",
    )
    defaults.update(kwargs)
    return ClipMetadata(**defaults)


def _make_http_error(status: int, reason: str = "", content: bytes = b"") -> HttpError:
    resp = MagicMock()
    resp.status = status
    resp.reason = reason
    return HttpError(resp=resp, content=content)


def _make_uploader(tmp_path: Path, creds=None) -> YouTubeUploader:
    uploader = YouTubeUploader(token_dir=tmp_path / "credentials")
    uploader._credentials = creds or MagicMock()
    return uploader


# ---------------------------------------------------------------------------
# upload() — success
# ---------------------------------------------------------------------------

class TestUploadSuccess:
    @patch("youcut.uploader.youtube.build")
    def test_returns_success_result_with_url(self, mock_build, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.return_value = (None, {"id": "abc123"})

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "success"
        assert result.url == "https://youtu.be/abc123"
        assert result.error is None

    @patch("youcut.uploader.youtube.build")
    def test_clip_index_propagated_to_result(self, mock_build, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.return_value = (None, {"id": "id42"})

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata(), clip_index=3)

        assert result.clip_index == 3

    @patch("youcut.uploader.youtube.build")
    def test_privacy_status_passed_to_api(self, mock_build, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.return_value = (None, {"id": "pub_id"})

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        uploader.upload(video, _make_metadata(), privacy="public")

        insert_call = mock_youtube.videos.return_value.insert
        body_used = insert_call.call_args.kwargs["body"]
        assert body_used["status"]["privacyStatus"] == "public"

    @patch("youcut.uploader.youtube.build")
    def test_next_chunk_called_until_response(self, mock_build, tmp_path):
        """next_chunk() may return (status, None) until upload completes."""
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = [
            (MagicMock(), None),
            (MagicMock(), None),
            (None, {"id": "xyz789"}),
        ]

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "success"
        assert result.url == "https://youtu.be/xyz789"


# ---------------------------------------------------------------------------
# upload() — 403 quotaExceeded
# ---------------------------------------------------------------------------

class TestUploadQuotaExceeded:
    @patch("youcut.uploader.youtube.build")
    def test_returns_failed_with_quota_message(self, mock_build, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = _make_http_error(
            403, content=b'{"error": {"errors": [{"reason": "quotaExceeded"}]}}'
        )

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "failed"
        assert result.url is None
        assert "quota" in result.error.lower()

    @patch("youcut.uploader.youtube.build")
    def test_does_not_retry_on_quota_exceeded(self, mock_build, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = _make_http_error(
            403, content=b'{"error":{"errors":[{"reason":"quotaExceeded"}]}}'
        )

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        uploader.upload(video, _make_metadata())

        # Only one call — no retry on quota errors
        assert mock_request.next_chunk.call_count == 1


# ---------------------------------------------------------------------------
# upload() — 5xx retry
# ---------------------------------------------------------------------------

class TestUpload5xxRetry:
    @patch("youcut.uploader.youtube.time.sleep")
    @patch("youcut.uploader.youtube.build")
    def test_retries_on_5xx_then_fails(self, mock_build, mock_sleep, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = _make_http_error(503, content=b"Service Unavailable")

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "failed"
        assert mock_sleep.call_count >= 2  # retried at least twice before final failure

    @patch("youcut.uploader.youtube.time.sleep")
    @patch("youcut.uploader.youtube.build")
    def test_retries_on_5xx_then_succeeds(self, mock_build, mock_sleep, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = [
            _make_http_error(500, content=b"Internal Server Error"),
            (None, {"id": "retry_ok"}),
        ]

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "success"
        assert result.url == "https://youtu.be/retry_ok"
        assert mock_sleep.call_count == 1

    @patch("youcut.uploader.youtube.time.sleep")
    @patch("youcut.uploader.youtube.build")
    def test_no_retry_on_4xx(self, mock_build, mock_sleep, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = _make_http_error(400, content=b"Bad Request")

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "failed"
        mock_sleep.assert_not_called()
        assert mock_request.next_chunk.call_count == 1


# ---------------------------------------------------------------------------
# upload() — timeout
# ---------------------------------------------------------------------------

class TestUploadTimeout:
    @patch("youcut.uploader.youtube.build")
    def test_returns_failed_on_socket_timeout(self, mock_build, tmp_path):
        uploader = _make_uploader(tmp_path)

        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_request = MagicMock()
        mock_youtube.videos.return_value.insert.return_value = mock_request
        mock_request.next_chunk.side_effect = socket.timeout("timed out")

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        result = uploader.upload(video, _make_metadata())

        assert result.status == "failed"
        assert result.url is None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


# ---------------------------------------------------------------------------
# authenticate() — token exists
# ---------------------------------------------------------------------------

class TestAuthenticateTokenExists:
    def test_does_not_open_browser_when_token_exists(self, tmp_path):
        token_dir = tmp_path / "credentials"
        token_dir.mkdir(mode=0o700)

        from youcut.uploader.auth import save_token
        save_token(
            "youtube",
            {
                "token": "access_token",
                "refresh_token": "refresh",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "client_id",
                "client_secret": "client_secret",
                "scopes": ["https://www.googleapis.com/auth/youtube.upload"],
            },
            token_dir,
        )

        uploader = YouTubeUploader(token_dir=token_dir)

        with patch("youcut.uploader.youtube.InstalledAppFlow") as mock_flow:
            uploader.authenticate()
            mock_flow.assert_not_called()

        assert uploader._credentials is not None

    def test_credentials_loaded_from_saved_token(self, tmp_path):
        token_dir = tmp_path / "credentials"
        token_dir.mkdir(mode=0o700)

        from youcut.uploader.auth import save_token
        token_data = {
            "token": "my_access_token",
            "refresh_token": "my_refresh_token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "my_client_id",
            "client_secret": "my_secret",
            "scopes": ["https://www.googleapis.com/auth/youtube.upload"],
        }
        save_token("youtube", token_data, token_dir)

        uploader = YouTubeUploader(token_dir=token_dir)
        uploader.authenticate()

        assert uploader._credentials.token == "my_access_token"
        assert uploader._credentials.client_id == "my_client_id"


# ---------------------------------------------------------------------------
# authenticate() — no token, starts OAuth flow
# ---------------------------------------------------------------------------

class TestAuthenticateNoToken:
    def test_opens_oauth_flow_when_no_token(self, tmp_path):
        token_dir = tmp_path / "credentials"
        secrets_file = tmp_path / "client_secrets.json"
        secrets_file.write_text("{}")

        uploader = YouTubeUploader(
            token_dir=token_dir,
            client_secrets_file=secrets_file,
        )

        mock_creds = MagicMock()
        mock_creds.token = "new_token"
        mock_creds.refresh_token = "new_refresh"
        mock_creds.token_uri = "https://oauth2.googleapis.com/token"
        mock_creds.client_id = "client"
        mock_creds.client_secret = "secret"
        mock_creds.scopes = ["https://www.googleapis.com/auth/youtube.upload"]

        with patch("youcut.uploader.youtube.InstalledAppFlow") as mock_flow_class:
            mock_flow = MagicMock()
            mock_flow_class.from_client_secrets_file.return_value = mock_flow
            mock_flow.run_local_server.return_value = mock_creds

            uploader.authenticate()

            mock_flow_class.from_client_secrets_file.assert_called_once()
            mock_flow.run_local_server.assert_called_once()

        assert uploader._credentials is mock_creds

        # Token must be persisted
        from youcut.uploader.auth import get_token
        saved = get_token("youtube", token_dir)
        assert saved is not None
        assert saved["token"] == "new_token"

    def test_raises_when_no_token_and_no_secrets_file(self, tmp_path):
        token_dir = tmp_path / "credentials"
        uploader = YouTubeUploader(token_dir=token_dir)

        with pytest.raises(RuntimeError, match="client_secrets_file"):
            uploader.authenticate()
