"""Tests for TikTokUploader using httpx.MockTransport."""
import math
import re
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from youcut.uploader.base import ClipMetadata
from youcut.uploader.tiktok import TikTokUploader, _CHUNK_SIZE

_ACCESS_TOKEN = "fake_tiktok_token"
_PUBLISH_ID = "publish_abc123"
_UPLOAD_URL = "https://open.tiktokapis.com/v2/upload/fake-session"
_VIDEO_ID = "7123456789012345678"


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
        title="Test TikTok",
        description="A great short video",
        hashtags=["#test", "#viral"],
        caption="Test TikTok\n\nA great short video\n\n#test #viral",
    )


def _make_uploader(
    token_dir: Path,
    transport: httpx.BaseTransport,
    polling_interval: float = 0.0,
    polling_max_retries: int = 10,
) -> TikTokUploader:
    uploader = TikTokUploader(
        token_dir=token_dir,
        transport=transport,
        polling_interval=polling_interval,
        polling_max_retries=polling_max_retries,
    )
    uploader._access_token = _ACCESS_TOKEN
    return uploader


def _sequential_transport(responses: list[tuple[int, dict]]) -> httpx.MockTransport:
    """MockTransport that returns responses in order."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise RuntimeError(f"Unexpected request: {request.method} {request.url}")
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def _init_response() -> tuple[int, dict]:
    return (
        200,
        {
            "data": {"publish_id": _PUBLISH_ID, "upload_url": _UPLOAD_URL},
            "error": {"code": "ok", "message": ""},
        },
    )


def _upload_response() -> tuple[int, dict]:
    return (200, {"data": {}, "error": {"code": "ok", "message": ""}})


def _status_complete_response() -> tuple[int, dict]:
    return (
        200,
        {
            "data": {
                "status": "PUBLISH_COMPLETE",
                "publish_id": _PUBLISH_ID,
                "publicaly_available_post_id": [_VIDEO_ID],
            },
            "error": {"code": "ok", "message": ""},
        },
    )


def _status_processing_response() -> tuple[int, dict]:
    return (
        200,
        {
            "data": {"status": "PROCESSING_UPLOAD", "publish_id": _PUBLISH_ID},
            "error": {"code": "ok", "message": ""},
        },
    )


class TestUploadSuccess:
    def test_returns_success_with_tiktok_url(self, token_dir, video_file, metadata):
        responses = [
            _init_response(),
            _upload_response(),
            _status_complete_response(),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=1)

        assert result.status == "success"
        assert result.url == f"https://www.tiktok.com/video/{_VIDEO_ID}"
        assert result.platform == "tiktok"
        assert result.clip_index == 1
        assert result.error is None

    def test_success_url_is_none_when_post_id_absent(self, token_dir, video_file, metadata):
        status_no_id = (
            200,
            {
                "data": {
                    "status": "PUBLISH_COMPLETE",
                    "publish_id": _PUBLISH_ID,
                    "publicaly_available_post_id": [],
                },
                "error": {"code": "ok", "message": ""},
            },
        )
        responses = [_init_response(), _upload_response(), status_no_id]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=2)

        assert result.status == "success"
        assert result.url is None


class TestPollingProcessingThenComplete:
    def test_polls_until_publish_complete(self, token_dir, video_file, metadata):
        responses = [
            _init_response(),
            _upload_response(),
            _status_processing_response(),  # first poll: still processing
            _status_complete_response(),    # second poll: done
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport, polling_interval=0.0)

        with patch("youcut.uploader.tiktok.time.sleep"):
            result = uploader.upload(video_file, metadata, clip_index=3)

        assert result.status == "success"
        assert result.url == f"https://www.tiktok.com/video/{_VIDEO_ID}"


class TestInitError:
    def test_http_4xx_on_init_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            (400, {"error": {"code": "invalid_param", "message": "Bad request"}}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=4)

        assert result.status == "failed"
        assert result.error is not None
        assert "400" in result.error

    def test_http_5xx_on_init_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            (500, {"error": {"code": "internal_error", "message": "Server error"}}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=5)

        assert result.status == "failed"
        assert "500" in result.error


class TestUploadChunkError:
    def test_http_5xx_on_chunk_upload_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            _init_response(),
            (500, {"error": {"code": "internal_error", "message": "Upload failed"}}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=6)

        assert result.status == "failed"
        assert result.error is not None
        assert "500" in result.error

    def test_http_4xx_on_chunk_upload_returns_failed(self, token_dir, video_file, metadata):
        responses = [
            _init_response(),
            (403, {"error": {"code": "access_denied", "message": "Forbidden"}}),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=7)

        assert result.status == "failed"
        assert "403" in result.error


class TestPollingRateLimit:
    def test_sleep_is_called_between_poll_attempts(self, token_dir, video_file, metadata):
        """Verify that time.sleep is called between polling requests."""
        responses = [
            _init_response(),
            _upload_response(),
            _status_processing_response(),
            _status_complete_response(),
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport, polling_interval=10.0)

        with patch("youcut.uploader.tiktok.time.sleep") as mock_sleep:
            result = uploader.upload(video_file, metadata, clip_index=8)

        assert result.status == "success"
        # sleep called once between attempt 0→1 (first attempt has no sleep)
        mock_sleep.assert_called_once_with(10.0)

    def test_polling_max_retries_respected(self, token_dir, video_file, metadata):
        """Verify that polling stops after max retries and returns failed."""
        # Always return processing status — never completes
        responses = [_init_response(), _upload_response()] + [
            _status_processing_response() for _ in range(3)
        ]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport, polling_interval=0.0, polling_max_retries=3)

        with patch("youcut.uploader.tiktok.time.sleep"):
            result = uploader.upload(video_file, metadata, clip_index=9)

        assert result.status == "pending"
        assert "inbox" in result.error.lower() or "processing" in result.error.lower()

    def test_no_sleep_on_first_poll_attempt(self, token_dir, video_file, metadata):
        """First polling attempt must not sleep (no wasted time before first check)."""
        responses = [_init_response(), _upload_response(), _status_complete_response()]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport, polling_interval=10.0)

        with patch("youcut.uploader.tiktok.time.sleep") as mock_sleep:
            result = uploader.upload(video_file, metadata, clip_index=10)

        assert result.status == "success"
        mock_sleep.assert_not_called()


class TestChunkSizeLimit:
    def test_small_file_sends_file_size_as_chunk_size(self, tmp_path, token_dir, metadata):
        """chunk_size in init must equal file_size when file fits in a single chunk."""
        video_path = tmp_path / "small_clip.mp4"
        video_path.write_bytes(b"x" * 1024)  # 1 KB

        received_init_body: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "init" in url:
                received_init_body.update(request.read() and __import__("json").loads(request.content))
                return httpx.Response(200, json={"data": {"publish_id": _PUBLISH_ID, "upload_url": "https://fake.upload/"}, "error": {"code": "ok", "message": ""}})
            if "fake.upload" in url:
                return httpx.Response(200, json={"data": {}})
            if "status" in url:
                return httpx.Response(200, json={"data": {"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": [_VIDEO_ID]}, "error": {"code": "ok", "message": ""}})
            return httpx.Response(404, text="not found")

        transport = httpx.MockTransport(handler)
        uploader = _make_uploader(token_dir, transport, polling_interval=0.0)
        uploader.upload(video_path, metadata, clip_index=99)

        source_info = received_init_body.get("source_info", {})
        assert source_info["chunk_size"] == 1024
        assert source_info["video_size"] == 1024
        assert source_info["total_chunk_count"] == 1

    def test_large_file_is_split_into_chunks_of_at_most_64mb(self, tmp_path, token_dir, metadata):
        """Verify that a file larger than 64 MB is split into ≤64 MB chunks."""
        chunk_size = _CHUNK_SIZE
        # 2.5 chunks worth of data
        file_size = int(chunk_size * 2.5)
        video_path = tmp_path / "big_clip.mp4"
        video_path.write_bytes(b"A" * file_size)

        expected_chunks = math.ceil(file_size / chunk_size)  # 3 chunks

        received_chunks: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "init" in url:
                return httpx.Response(
                    200,
                    json={
                        "data": {"publish_id": _PUBLISH_ID, "upload_url": "https://fake.upload/"},
                        "error": {"code": "ok", "message": ""},
                    },
                )
            if "fake.upload" in url and request.method == "PUT":
                content_range = request.headers.get("content-range", "")
                # parse "bytes start-end/total"
                _, range_part = content_range.split(" ", 1)
                range_def, _ = range_part.split("/", 1)
                start, end = range_def.split("-")
                chunk_len = int(end) - int(start) + 1
                assert chunk_len <= chunk_size, f"Chunk too large: {chunk_len} > {chunk_size}"
                received_chunks.append(chunk_len)
                return httpx.Response(200, json={"data": {}})
            if "status" in url:
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "status": "PUBLISH_COMPLETE",
                            "publicaly_available_post_id": [_VIDEO_ID],
                        },
                        "error": {"code": "ok", "message": ""},
                    },
                )
            return httpx.Response(404, text="not found")

        transport = httpx.MockTransport(handler)
        uploader = _make_uploader(token_dir, transport, polling_interval=0.0)

        result = uploader.upload(video_path, metadata, clip_index=11)

        assert result.status == "success"
        assert len(received_chunks) == expected_chunks
        for chunk_len in received_chunks:
            assert chunk_len <= chunk_size


class TestPrivateVideoWarning:
    def test_warning_logged_on_every_upload(self, token_dir, video_file, metadata, caplog):
        import logging

        responses = [_init_response(), _upload_response(), _status_complete_response()]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        with caplog.at_level(logging.WARNING, logger="youcut.uploader.tiktok"):
            uploader.upload(video_file, metadata, clip_index=12)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("draft" in msg.lower() or "inbox" in msg.lower() for msg in warning_messages), (
            "Expected a warning about draft/inbox, got: " + str(warning_messages)
        )

    def test_warning_logged_even_on_failed_upload(self, token_dir, video_file, metadata, caplog):
        import logging

        responses = [(400, {"error": {"code": "bad", "message": "fail"}})]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        with caplog.at_level(logging.WARNING, logger="youcut.uploader.tiktok"):
            uploader.upload(video_file, metadata, clip_index=13)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("draft" in msg.lower() or "inbox" in msg.lower() for msg in warning_messages)


class TestAuthenticate:
    def test_loads_token_from_saved_credentials(self, token_dir):
        from youcut.uploader.auth import save_token

        save_token(
            "tiktok",
            {
                "access_token": "saved_tok",
                "refresh_token": "refresh_tok",
                "expires_at": "2999-01-01T00:00:00+00:00",
                "client_key": "client_key_123",
            },
            token_dir,
        )

        uploader = TikTokUploader(token_dir=token_dir)
        uploader.authenticate()

        assert uploader._access_token == "saved_tok"

    def test_starts_pkce_flow_when_no_saved_credentials(self, token_dir):
        uploader = TikTokUploader(token_dir=token_dir)
        oauth_token = {
            "access_token": "oauth_tok",
            "refresh_token": "refresh_tok",
            "expires_at": "2999-01-01T00:00:00+00:00",
            "client_key": "client_key_123",
        }

        with patch.object(uploader, "_run_pkce_oauth_flow", return_value=oauth_token) as mock_flow:
            uploader.authenticate()

        mock_flow.assert_called_once_with()
        assert uploader._access_token == "oauth_tok"

        from youcut.uploader.auth import get_token

        saved = get_token("tiktok", token_dir)
        assert saved is not None
        assert saved["access_token"] == "oauth_tok"
        assert saved["refresh_token"] == "refresh_tok"

    def test_refreshes_expired_token_before_upload(self, token_dir):
        from youcut.uploader.auth import get_token, save_token

        save_token(
            "tiktok",
            {
                "access_token": "expired_tok",
                "refresh_token": "refresh_tok",
                "expires_at": "2000-01-01T00:00:00+00:00",
                "client_key": "client_key_123",
            },
            token_dir,
        )
        uploader = TikTokUploader(token_dir=token_dir)
        refreshed_token = {
            "access_token": "fresh_tok",
            "refresh_token": "refresh_tok_new",
            "expires_at": "2999-01-01T00:00:00+00:00",
            "client_key": "client_key_123",
        }

        with patch.object(uploader, "_refresh_access_token", return_value=refreshed_token) as mock_refresh:
            uploader.authenticate()

        mock_refresh.assert_called_once()
        assert uploader._access_token == "fresh_tok"
        saved = get_token("tiktok", token_dir)
        assert saved is not None
        assert saved["access_token"] == "fresh_tok"
        assert saved["refresh_token"] == "refresh_tok_new"

    def test_falls_back_to_pkce_when_refresh_fails(self, token_dir):
        from youcut.uploader.auth import save_token

        save_token(
            "tiktok",
            {
                "access_token": "expired_tok",
                "refresh_token": "refresh_tok",
                "expires_at": "2000-01-01T00:00:00+00:00",
                "client_key": "client_key_123",
            },
            token_dir,
        )
        uploader = TikTokUploader(token_dir=token_dir)
        oauth_token = {
            "access_token": "oauth_tok",
            "refresh_token": "oauth_refresh",
            "expires_at": "2999-01-01T00:00:00+00:00",
            "client_key": "client_key_123",
        }

        with patch.object(
            uploader,
            "_refresh_access_token",
            side_effect=RuntimeError("refresh failed"),
        ) as mock_refresh:
            with patch.object(uploader, "_run_pkce_oauth_flow", return_value=oauth_token) as mock_flow:
                uploader.authenticate()

        mock_refresh.assert_called_once()
        mock_flow.assert_called_once_with()
        assert uploader._access_token == "oauth_tok"

    def test_raises_when_pkce_flow_cannot_start(self, token_dir):
        uploader = TikTokUploader(token_dir=token_dir)

        with patch.object(
            uploader,
            "_run_pkce_oauth_flow",
            side_effect=RuntimeError(
                "TikTok OAuth requires TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET"
            ),
        ):
            with pytest.raises(RuntimeError, match="TIKTOK_CLIENT_SECRET"):
                uploader.authenticate()


class TestTokenParsing:
    def test_accepts_double_nested_data_payload(self, token_dir):
        uploader = TikTokUploader(token_dir=token_dir)
        response = httpx.Response(
            200,
            json={
                "data": {
                    "data": {
                        "access_token": "oauth_tok",
                        "refresh_token": "refresh_tok",
                        "expires_in": 3600,
                    }
                }
            },
        )

        parsed = uploader._parse_token_response(response, "TikTok OAuth token exchange failed")

        assert parsed["access_token"] == "oauth_tok"
        assert parsed["refresh_token"] == "refresh_tok"
        assert "expires_at" in parsed

    def test_surfaces_api_error_when_access_token_is_missing(self, token_dir):
        uploader = TikTokUploader(token_dir=token_dir)
        response = httpx.Response(
            200,
            json={
                "error": {
                    "code": "invalid_client",
                    "message": "Client key is not approved for this scope.",
                }
            },
        )

        with pytest.raises(RuntimeError, match="invalid_client: Client key is not approved for this scope."):
            uploader._parse_token_response(response, "TikTok OAuth token exchange failed")


class TestTokenRequests:
    def test_exchange_code_for_token_sends_client_secret(self, token_dir, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content.decode()
            return httpx.Response(200, json={"access_token": "oauth_tok", "expires_in": 3600})

        uploader = TikTokUploader(token_dir=token_dir, transport=httpx.MockTransport(handler))

        uploader._exchange_code_for_token(
            client_key="client_key_123",
            client_secret="client_secret_456",
            redirect_uri="http://127.0.0.1:8765/callback",
            code="auth_code_123",
            code_verifier="verifier_123",
        )

        assert "client_key=client_key_123" in seen["body"]
        assert "client_secret=client_secret_456" in seen["body"]
        assert "grant_type=authorization_code" in seen["body"]

    def test_refresh_access_token_sends_client_secret(self, token_dir, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content.decode()
            return httpx.Response(200, json={"access_token": "fresh_tok", "expires_in": 3600})

        monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "client_secret_456")
        uploader = TikTokUploader(token_dir=token_dir, transport=httpx.MockTransport(handler))

        refreshed = uploader._refresh_access_token(
            {
                "client_key": "client_key_123",
                "refresh_token": "refresh_tok_123",
            }
        )

        assert refreshed["access_token"] == "fresh_tok"
        assert "client_key=client_key_123" in seen["body"]
        assert "client_secret=client_secret_456" in seen["body"]
        assert "grant_type=refresh_token" in seen["body"]


class TestPkceHelpers:
    def test_code_verifier_matches_tiktok_sdk_constraints(self, token_dir):
        uploader = TikTokUploader(token_dir=token_dir)

        verifier = uploader._build_code_verifier()

        assert len(verifier) == 43
        assert re.fullmatch(r"[A-Za-z0-9._-]{43}", verifier)

    def test_code_challenge_uses_sha256_hex_digest(self, token_dir):
        uploader = TikTokUploader(token_dir=token_dir)

        challenge = uploader._build_code_challenge("abcABC123-._xyz")

        assert challenge == "a182fa409909d6c15763daf57df4bc9585c560e0810135111b7925744c1876fc"


class TestPollingPublishFailed:
    def test_tiktok_failed_status_returns_failed_result(self, token_dir, video_file, metadata):
        status_failed = (
            200,
            {
                "data": {
                    "status": "FAILED",
                    "publish_id": _PUBLISH_ID,
                    "fail_reason": "VIDEO_TOO_SHORT",
                },
                "error": {"code": "ok", "message": ""},
            },
        )
        responses = [_init_response(), _upload_response(), status_failed]
        transport = _sequential_transport(responses)
        uploader = _make_uploader(token_dir, transport)

        result = uploader.upload(video_file, metadata, clip_index=14)

        assert result.status == "failed"
        assert "VIDEO_TOO_SHORT" in result.error
