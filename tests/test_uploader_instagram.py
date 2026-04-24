"""Tests for InstagramUploader using httpx.MockTransport."""
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from youcut.uploader.base import ClipMetadata
from youcut.uploader.instagram import InstagramUploader

_IG_USER_ID = "ig_user_123"
_ACCESS_TOKEN = "fake_token_abc"
_CONTAINER_ID = "container_456"
_MEDIA_ID = "reel_xyz789"


@pytest.fixture
def token_dir(tmp_path: Path) -> Path:
    return tmp_path / "credentials"


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    p = tmp_path / "clip_01.mp4"
    p.write_bytes(b"x" * 1024)  # 1 KB fake video
    return p


@pytest.fixture
def metadata() -> ClipMetadata:
    return ClipMetadata(
        title="Test Reel",
        description="A great reel",
        hashtags=["#test", "#viral"],
        caption="A great reel\n\n#test #viral",
    )


def _make_uploader(
    token_dir: Path,
    transport: httpx.BaseTransport,
    polling_interval: float = 0.0,
    polling_timeout: float = 60.0,
) -> InstagramUploader:
    uploader = InstagramUploader(
        token_dir=token_dir,
        transport=transport,
        polling_interval=polling_interval,
        polling_timeout=polling_timeout,
    )
    uploader._access_token = _ACCESS_TOKEN
    uploader._ig_user_id = _IG_USER_ID
    return uploader


def _sequential_transport(responses: list[tuple[int, dict]]) -> httpx.MockTransport:
    """MockTransport that returns responses in sequence."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise RuntimeError(f"Unexpected request: {request.method} {request.url}")
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def _success_responses() -> list[tuple[int, dict]]:
    return [
        # 1. account type check
        (200, {"account_type": "BUSINESS", "id": _IG_USER_ID}),
        # 2. create container
        (200, {"id": _CONTAINER_ID}),
        # 3. chunked upload
        (200, {"success": True, "message": ""}),
        # 4. poll status
        (200, {"status_code": "FINISHED", "id": _CONTAINER_ID}),
        # 5. publish
        (200, {"id": _MEDIA_ID}),
    ]


class TestUploadSuccess:
    def test_returns_success_with_reel_url(self, token_dir, video_file, metadata):
        transport = _sequential_transport(_success_responses())
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=1)

        assert result.status == "success"
        assert result.url == f"https://www.instagram.com/reel/{_MEDIA_ID}/"
        assert result.platform == "instagram"
        assert result.clip_index == 1
        assert result.error is None


class TestPollingInProgressThenFinished:
    def test_polls_until_finished(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "BUSINESS", "id": _IG_USER_ID}),
            (200, {"id": _CONTAINER_ID}),
            (200, {"success": True}),
            # first poll: IN_PROGRESS
            (200, {"status_code": "IN_PROGRESS", "id": _CONTAINER_ID}),
            # second poll: FINISHED
            (200, {"status_code": "FINISHED", "id": _CONTAINER_ID}),
            (200, {"id": _MEDIA_ID}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport, polling_interval=0.0, polling_timeout=30.0)

        with patch("youcut.uploader.instagram.time.sleep"):
            result = uploader.upload(video_file, metadata, clip_index=2)

        assert result.status == "success"
        assert result.url == f"https://www.instagram.com/reel/{_MEDIA_ID}/"


class TestPollingError:
    def test_container_error_state_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "BUSINESS", "id": _IG_USER_ID}),
            (200, {"id": _CONTAINER_ID}),
            (200, {"success": True}),
            (200, {"status_code": "ERROR", "id": _CONTAINER_ID}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=3)

        assert result.status == "failed"
        assert result.error is not None
        assert "error" in result.error.lower() or "Container" in result.error


class TestPollingTimeout:
    def test_timeout_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "BUSINESS", "id": _IG_USER_ID}),
            (200, {"id": _CONTAINER_ID}),
            (200, {"success": True}),
            # polling will not be reached — timeout fires immediately
        ]
        transport = _sequential_transport(responses)
        # polling_timeout=-1 ensures deadline is already in the past
        uploader = _make_uploader(token_dir, transport, polling_interval=0.0, polling_timeout=-1.0)

        result = uploader.upload(video_file, metadata, clip_index=4)

        assert result.status == "failed"
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()


class TestContainerCreationError:
    def test_http_4xx_on_create_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "BUSINESS", "id": _IG_USER_ID}),
            # 400 on container creation
            (400, {"error": {"message": "Invalid parameter", "code": 100}}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=5)

        assert result.status == "failed"
        assert result.error is not None
        assert "400" in result.error


class TestChunkedUploadError:
    def test_http_5xx_on_upload_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "BUSINESS", "id": _IG_USER_ID}),
            (200, {"id": _CONTAINER_ID}),
            # 500 on rupload
            (500, {"error": {"message": "Internal Server Error"}}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=6)

        assert result.status == "failed"
        assert result.error is not None
        assert "500" in result.error


class TestNonBusinessAccount:
    def test_personal_account_returns_descriptive_error(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "PERSONAL", "id": _IG_USER_ID}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=7)

        assert result.status == "failed"
        assert result.error is not None
        assert "BUSINESS" in result.error or "MEDIA_CREATOR" in result.error
        assert "PERSONAL" in result.error

    def test_media_creator_account_is_eligible(self, token_dir, video_file, metadata):
        responses = [
            (200, {"account_type": "MEDIA_CREATOR", "id": _IG_USER_ID}),
            (200, {"id": _CONTAINER_ID}),
            (200, {"success": True}),
            (200, {"status_code": "FINISHED", "id": _CONTAINER_ID}),
            (200, {"id": _MEDIA_ID}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=8)

        assert result.status == "success"


class TestAuthenticate:
    def test_loads_token_from_saved_credentials(self, token_dir, tmp_path):
        from youcut.uploader.auth import save_token

        save_token("instagram", {"access_token": "saved_tok", "ig_user_id": "saved_uid"}, token_dir)

        uploader = InstagramUploader(token_dir=token_dir)
        uploader.authenticate()

        assert uploader._access_token == "saved_tok"
        assert uploader._ig_user_id == "saved_uid"

    def test_raises_when_no_credentials_and_no_env(self, token_dir, monkeypatch):
        monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("INSTAGRAM_USER_ID", raising=False)

        uploader = InstagramUploader(token_dir=token_dir)

        with patch.object(
            uploader,
            "_run_oauth_flow",
            side_effect=RuntimeError("Instagram OAuth requires INSTAGRAM_APP_ID"),
        ):
            with pytest.raises(RuntimeError, match="INSTAGRAM_APP_ID"):
                uploader.authenticate()

    def test_loads_from_env_vars_and_saves(self, token_dir, monkeypatch):
        monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "env_tok")
        monkeypatch.setenv("INSTAGRAM_USER_ID", "env_uid")

        uploader = InstagramUploader(token_dir=token_dir)
        uploader.authenticate()

        assert uploader._access_token == "env_tok"
        assert uploader._ig_user_id == "env_uid"

        from youcut.uploader.auth import get_token

        saved = get_token("instagram", token_dir)
        assert saved is not None
        assert saved["access_token"] == "env_tok"

    def test_starts_oauth_flow_when_no_saved_credentials(self, token_dir):
        uploader = InstagramUploader(token_dir=token_dir)

        with patch.object(
            uploader,
            "_run_oauth_flow",
            return_value=("oauth_tok", "oauth_uid"),
        ) as mock_oauth:
            uploader.authenticate()

        mock_oauth.assert_called_once_with()
        assert uploader._access_token == "oauth_tok"
        assert uploader._ig_user_id == "oauth_uid"

        from youcut.uploader.auth import get_token

        saved = get_token("instagram", token_dir)
        assert saved is not None
        assert saved["access_token"] == "oauth_tok"
        assert saved["ig_user_id"] == "oauth_uid"
