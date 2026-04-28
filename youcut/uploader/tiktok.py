import hashlib
import logging
import math
import os
import subprocess
import secrets
import time
import webbrowser
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .auth import get_token, revoke_token, save_token
from .base import ClipMetadata, UploadResult, Uploader
from .metadata import apply_platform_limits

logger = logging.getLogger(__name__)

_PLATFORM = "tiktok"
_API_BASE = "https://open.tiktokapis.com"
_AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
_TOKEN_URL = f"{_API_BASE}/v2/oauth/token/"
_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB max per chunk per TikTok spec
_POLLING_INTERVAL = 10.0  # 6 req/min → 1 req every 10 s
_POLLING_MAX_RETRIES = 60  # ~10 minutes total
_AUTH_TIMEOUT = 300.0
_ACCESS_TOKEN_SKEW = timedelta(seconds=30)
_PKCE_VERIFIER_LENGTH = 43
_PKCE_VERIFIER_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
_POST_MODE_DRAFT = "draft"
_POST_MODE_DIRECT = "direct"
_VALID_POST_MODES = frozenset({_POST_MODE_DRAFT, _POST_MODE_DIRECT})
_VALID_PRIVACY_LEVELS = frozenset(
    {"PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "SELF_ONLY"}
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value (true/false).")


def _extract_response_error(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if code in (None, "", "ok"):
        return None
    message = error.get("message") or "unknown error"
    return f"{code}: {message}"


def _format_post_init_error(error: str, post_mode: str) -> str:
    if post_mode == _POST_MODE_DRAFT and "spam_risk_too_many_pending_share" in error:
        return (
            "Init failed: TikTok bloqueou novos envios para a inbox porque ha rascunhos/compartilhamentos "
            "pendentes demais nessa conta. Conta privada por si so nao muda o fluxo de inbox. "
            "Para publicar direto pela API, configure TIKTOK_POST_MODE=direct e refaca "
            "`youcut auth login --platform tiktok` para conceder o escopo video.publish. "
            "Se quiser continuar no modo draft, limpe os pendentes na inbox do TikTok antes de tentar de novo."
        )
    return f"Init failed: {error}"


def _get_video_duration_seconds(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    output = result.stdout.strip()
    if not output:
        return None
    try:
        return float(output)
    except ValueError:
        return None


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server_version = "YouCutTikTokOAuth/1.0"

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        query = parse_qs(urlparse(self.path).query)
        server.auth_query = query
        server.auth_path = self.path
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h1>Authentication complete.</h1>"
            b"<p>You can close this window and return to YouCut.</p></body></html>"
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _OAuthHTTPServer(HTTPServer):
    auth_query: dict[str, list[str]] | None
    auth_path: str | None

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _OAuthCallbackHandler)
        self.auth_query = None
        self.auth_path = None


class TikTokUploader(Uploader):
    def __init__(
        self,
        token_dir: Path,
        transport: httpx.BaseTransport | None = None,
        polling_interval: float = _POLLING_INTERVAL,
        polling_max_retries: int = _POLLING_MAX_RETRIES,
        post_mode: str | None = None,
        privacy_level: str | None = None,
        disable_comment: bool | None = None,
        disable_duet: bool | None = None,
        disable_stitch: bool | None = None,
    ) -> None:
        super().__init__(token_dir)
        self._access_token: str | None = None
        self._transport = transport
        self._polling_interval = polling_interval
        self._polling_max_retries = polling_max_retries
        resolved_post_mode = (post_mode or os.getenv("TIKTOK_POST_MODE", _POST_MODE_DRAFT)).strip().lower()
        if resolved_post_mode not in _VALID_POST_MODES:
            raise ValueError(
                f"TIKTOK_POST_MODE must be one of: {', '.join(sorted(_VALID_POST_MODES))}."
            )
        resolved_privacy_level = (
            privacy_level or os.getenv("TIKTOK_PRIVACY_LEVEL", "SELF_ONLY")
        ).strip().upper()
        if resolved_privacy_level not in _VALID_PRIVACY_LEVELS:
            raise ValueError(
                "TIKTOK_PRIVACY_LEVEL must be one of: "
                f"{', '.join(sorted(_VALID_PRIVACY_LEVELS))}."
            )
        self._post_mode = resolved_post_mode
        self._privacy_level = resolved_privacy_level
        self._disable_comment = (
            _parse_env_bool("TIKTOK_DISABLE_COMMENT", False)
            if disable_comment is None
            else disable_comment
        )
        self._disable_duet = (
            _parse_env_bool("TIKTOK_DISABLE_DUET", False)
            if disable_duet is None
            else disable_duet
        )
        self._disable_stitch = (
            _parse_env_bool("TIKTOK_DISABLE_STITCH", False)
            if disable_stitch is None
            else disable_stitch
        )

    @property
    def platform_name(self) -> str:
        return _PLATFORM

    def _make_client(self) -> httpx.Client:
        kwargs: dict = {"timeout": 60.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _load_saved_token(self) -> dict | None:
        token = get_token(_PLATFORM, self.token_dir)
        return token if isinstance(token, dict) else None

    def _token_is_valid(self, token: dict) -> bool:
        access_token = token.get("access_token")
        expires_at = token.get("expires_at")
        if not access_token:
            return False
        if not expires_at:
            return True
        try:
            expiry = datetime.fromisoformat(expires_at)
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return expiry > _utcnow() + _ACCESS_TOKEN_SKEW

    def _set_active_token(self, token: dict) -> None:
        self._access_token = token["access_token"]

    def _build_code_verifier(self) -> str:
        return "".join(secrets.choice(_PKCE_VERIFIER_CHARSET) for _ in range(_PKCE_VERIFIER_LENGTH))

    def _build_code_challenge(self, code_verifier: str) -> str:
        return hashlib.sha256(code_verifier.encode("ascii")).hexdigest()

    _CALLBACK_PORT = 8765

    def _start_callback_server(self) -> tuple[_OAuthHTTPServer, str]:
        override = os.getenv("TIKTOK_REDIRECT_URI")
        if override:
            redirect_uri = override
            port = int(urlparse(override).port or self._CALLBACK_PORT)
        else:
            port = self._CALLBACK_PORT
            redirect_uri = f"http://127.0.0.1:{port}/callback"
        server = _OAuthHTTPServer(("127.0.0.1", port))
        return server, redirect_uri

    def _wait_for_callback(
        self,
        server: _OAuthHTTPServer,
        timeout: float = _AUTH_TIMEOUT,
    ) -> tuple[str | None, str | None, str | None]:
        deadline = time.monotonic() + timeout
        server.timeout = 0.5

        while time.monotonic() < deadline:
            server.handle_request()
            if server.auth_query is not None:
                code = server.auth_query.get("code", [None])[0]
                state = server.auth_query.get("state", [None])[0]
                error = server.auth_query.get("error", [None])[0]
                return code, state, error

        raise RuntimeError("TikTok OAuth callback timed out.")

    def _build_authorization_url(
        self,
        client_key: str,
        redirect_uri: str,
        state: str,
        code_challenge: str,
    ) -> str:
        query = urlencode(
            {
                "client_key": client_key,
                "scope": self._oauth_scope(),
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{_AUTH_URL}?{query}"

    def _oauth_scope(self) -> str:
        scopes = ["user.info.basic", "video.upload"]
        if self._post_mode == _POST_MODE_DIRECT:
            scopes.append("video.publish")
        return ",".join(scopes)

    def _token_has_required_scope(self, token: dict) -> bool:
        if self._post_mode != _POST_MODE_DIRECT:
            return True
        granted = token.get("scope") or token.get("scopes")
        if not granted:
            return True
        if isinstance(granted, str):
            granted_scopes = {scope.strip() for scope in granted.replace(",", " ").split() if scope.strip()}
        elif isinstance(granted, list):
            granted_scopes = {str(scope).strip() for scope in granted if str(scope).strip()}
        else:
            return True
        return "video.publish" in granted_scopes

    def _exchange_code_for_token(
        self,
        *,
        client_key: str,
        client_secret: str,
        redirect_uri: str,
        code: str,
        code_verifier: str,
    ) -> dict:
        with self._make_client() as client:
            response = client.post(
                _TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_key": client_key,
                    "client_secret": client_secret,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "code_verifier": code_verifier,
                },
            )
        return self._parse_token_response(response, "TikTok OAuth token exchange failed")

    def _refresh_access_token(self, token: dict) -> dict:
        refresh_token = token.get("refresh_token")
        client_key = token.get("client_key") or os.getenv("TIKTOK_CLIENT_KEY")
        client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
        if not refresh_token or not client_key or not client_secret:
            raise RuntimeError("Saved TikTok credentials cannot be refreshed.")

        with self._make_client() as client:
            response = client.post(
                _TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_key": client_key,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )

        refreshed = self._parse_token_response(response, "TikTok token refresh failed")
        refreshed.setdefault("refresh_token", refresh_token)
        refreshed.setdefault(
            "refresh_expires_at",
            token.get("refresh_expires_at"),
        )
        refreshed["client_key"] = client_key
        return refreshed

    def _parse_token_response(self, response: httpx.Response, context: str) -> dict:
        if response.status_code != 200:
            raise RuntimeError(f"{context}: HTTP {response.status_code}: {response.text}")

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"{context}: unexpected response body: {payload!r}")

        data = payload
        # TikTok token responses may wrap the token payload under one or more `data` keys.
        while isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = data["data"]

        access_token = data.get("access_token")
        if not access_token:
            error = payload.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message")
                if code or message:
                    raise RuntimeError(
                        f"{context}: {code or 'unknown_error'}: {message or 'missing error message'}"
                    )

            message = payload.get("message")
            if message:
                raise RuntimeError(f"{context}: {message}")

            raise RuntimeError(f"{context}: response missing access_token: {payload}")

        normalized = dict(data)
        normalized["access_token"] = access_token

        expires_in = data.get("expires_in")
        if expires_in is not None:
            normalized["expires_at"] = (_utcnow() + timedelta(seconds=int(expires_in))).isoformat()

        refresh_token = data.get("refresh_token")
        refresh_expires_in = data.get("refresh_expires_in")
        if refresh_token is not None:
            normalized["refresh_token"] = refresh_token
        if refresh_expires_in is not None:
            normalized["refresh_expires_at"] = (
                _utcnow() + timedelta(seconds=int(refresh_expires_in))
            ).isoformat()

        return normalized

    def _run_pkce_oauth_flow(self) -> dict:
        client_key = os.getenv("TIKTOK_CLIENT_KEY")
        client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
        if not client_key or not client_secret:
            raise RuntimeError(
                "TikTok OAuth requires TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET to be configured before login."
            )

        code_verifier = self._build_code_verifier()
        code_challenge = self._build_code_challenge(code_verifier)
        state = secrets.token_urlsafe(24)
        server, redirect_uri = self._start_callback_server()
        auth_url = self._build_authorization_url(
            client_key=client_key,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
        )

        logger.info("Starting TikTok OAuth PKCE flow. Browser will open for login.")
        browser_thread = Thread(target=webbrowser.open, args=(auth_url,), kwargs={"new": 1}, daemon=True)
        browser_thread.start()

        try:
            code, returned_state, error = self._wait_for_callback(server)
        finally:
            server.server_close()

        if error:
            raise RuntimeError(f"TikTok authorization failed: {error}")
        if returned_state != state:
            raise RuntimeError("TikTok OAuth state mismatch.")
        if not code:
            raise RuntimeError("TikTok OAuth callback returned no authorization code.")

        token = self._exchange_code_for_token(
            client_key=client_key,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            code=code,
            code_verifier=code_verifier,
        )
        token["client_key"] = client_key
        return token

    def authenticate(self) -> None:
        token = self._load_saved_token()
        if token and self._token_is_valid(token) and self._token_has_required_scope(token):
            logger.debug("Reusing saved TikTok credentials.")
            self._set_active_token(token)
            return

        if token and token.get("refresh_token"):
            try:
                refreshed = self._refresh_access_token(token)
            except RuntimeError as exc:
                logger.warning("TikTok token refresh failed, restarting OAuth flow: %s", exc)
                revoke_token(_PLATFORM, self.token_dir)
            else:
                save_token(_PLATFORM, refreshed, self.token_dir)
                self._set_active_token(refreshed)
                logger.info("TikTok access token refreshed.")
                return

        token = self._run_pkce_oauth_flow()
        save_token(_PLATFORM, token, self.token_dir)
        self._set_active_token(token)
        logger.info("TikTok OAuth credentials saved.")

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        json_body: dict | None = None,
    ) -> tuple[dict | None, str | None]:
        with self._make_client() as client:
            try:
                resp = client.request(
                    method,
                    url,
                    headers={
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": "application/json; charset=UTF-8",
                    },
                    json=json_body,
                )
            except httpx.HTTPError as exc:
                return None, str(exc)

        try:
            payload = resp.json()
        except ValueError:
            payload = None

        api_error = _extract_response_error(payload)
        if resp.status_code != 200:
            message = api_error or resp.text
            return None, f"HTTP {resp.status_code}: {message}"
        if api_error:
            return None, api_error
        if not isinstance(payload, dict):
            return None, f"Unexpected response body: {resp.text}"
        return payload, None

    def _fetch_creator_info(self) -> tuple[dict | None, str | None]:
        payload, error = self._request_json(
            method="POST",
            url=f"{_API_BASE}/v2/post/publish/creator_info/query/",
        )
        if error:
            return None, f"Creator info query failed: {error}"
        data = payload.get("data")
        if not isinstance(data, dict):
            return None, "Creator info query failed: response missing data payload."
        return data, None

    def _build_post_info(self, metadata: ClipMetadata, creator_info: dict, video_path: Path) -> dict:
        privacy_options = creator_info.get("privacy_level_options")
        if not isinstance(privacy_options, list) or not privacy_options:
            raise RuntimeError("TikTok creator info did not return privacy_level_options.")
        if self._privacy_level not in privacy_options:
            raise RuntimeError(
                "TikTok privacy level "
                f"{self._privacy_level} is not allowed for this creator/app. "
                f"Available options: {', '.join(str(option) for option in privacy_options)}"
            )

        duration_limit = creator_info.get("max_video_post_duration_sec")
        duration_seconds = _get_video_duration_seconds(video_path)
        if isinstance(duration_limit, int) and duration_seconds is not None and duration_seconds > duration_limit:
            raise RuntimeError(
                f"TikTok allows at most {duration_limit}s for this creator, but the clip has about {duration_seconds:.1f}s."
            )

        disable_comment = self._disable_comment or bool(creator_info.get("comment_disabled"))
        disable_duet = self._disable_duet or bool(creator_info.get("duet_disabled"))
        disable_stitch = self._disable_stitch or bool(creator_info.get("stitch_disabled"))

        return {
            "title": metadata.caption,
            "privacy_level": self._privacy_level,
            "disable_comment": disable_comment,
            "disable_duet": disable_duet,
            "disable_stitch": disable_stitch,
        }

    def _init_post(
        self, video_path: Path, metadata: ClipMetadata
    ) -> tuple[str | None, str | None, str | None]:
        """Initialize TikTok post. Returns (publish_id, upload_url, error_msg)."""
        file_size = video_path.stat().st_size
        total_chunk_count = max(1, math.ceil(file_size / _CHUNK_SIZE))
        chunk_size = file_size if total_chunk_count == 1 else _CHUNK_SIZE
        init_endpoint = f"{_API_BASE}/v2/post/publish/inbox/video/init/"
        payload: dict[str, object] = {
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": file_size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunk_count,
            },
        }

        if self._post_mode == _POST_MODE_DIRECT:
            init_endpoint = f"{_API_BASE}/v2/post/publish/video/init/"
            creator_info, error = self._fetch_creator_info()
            if error:
                return None, None, error
            try:
                payload["post_info"] = self._build_post_info(metadata, creator_info, video_path)
            except RuntimeError as exc:
                return None, None, str(exc)

        response_payload, error = self._request_json(
            method="POST",
            url=init_endpoint,
            json_body=payload,
        )
        if error:
            return None, None, _format_post_init_error(error, self._post_mode)

        data = response_payload.get("data", {})
        publish_id = data.get("publish_id")
        upload_url = data.get("upload_url")

        if not publish_id or not upload_url:
            return None, None, f"Init response missing publish_id or upload_url: {response_payload}"

        logger.debug("TikTok post initialized: publish_id=%s", publish_id)
        return publish_id, upload_url, None

    def _upload_chunks(self, video_path: Path, upload_url: str) -> str | None:
        """Upload video in chunks via PUT to upload_url. Returns error or None."""
        file_size = video_path.stat().st_size
        offset = 0
        chunk_index = 0

        with video_path.open("rb") as f, self._make_client() as client:
            while offset < file_size:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break

                chunk_len = len(chunk)
                end = offset + chunk_len - 1

                try:
                    resp = client.put(
                        upload_url,
                        headers={
                            "Content-Range": f"bytes {offset}-{end}/{file_size}",
                            "Content-Type": "video/mp4",
                            "Content-Length": str(chunk_len),
                        },
                        content=chunk,
                    )
                except httpx.HTTPError as exc:
                    return f"Chunk {chunk_index} upload failed at offset {offset}: {exc}"

                if resp.status_code not in (200, 201, 206):
                    return f"Chunk {chunk_index} upload failed: HTTP {resp.status_code}: {resp.text}"

                offset += chunk_len
                chunk_index += 1
                logger.debug("Uploaded chunk %d (%d/%d bytes)", chunk_index, offset, file_size)

        return None

    def _poll_status(
        self, publish_id: str
    ) -> tuple[str, str | None, str | None]:
        """Poll publish status until complete, failed, or max retries.

        Sleeps _polling_interval before each attempt (except the first) to
        respect TikTok's 6 req/min rate limit on this endpoint.

        Returns (status, video_url, error_msg).
        """
        for attempt in range(self._polling_max_retries):
            if attempt > 0:
                time.sleep(self._polling_interval)

            with self._make_client() as client:
                try:
                    resp = client.post(
                        f"{_API_BASE}/v2/post/publish/status/fetch/",
                        headers={
                            "Authorization": f"Bearer {self._access_token}",
                            "Content-Type": "application/json; charset=UTF-8",
                        },
                        json={"publish_id": publish_id},
                    )
                except httpx.HTTPError as exc:
                    return "FAILED", None, f"Status polling failed: {exc}"

            if resp.status_code != 200:
                return (
                    "FAILED",
                    None,
                    f"Status polling failed: HTTP {resp.status_code}: {resp.text}",
                )

            data = resp.json().get("data", {})
            status = data.get("status", "")
            logger.debug("TikTok publish_id=%s status: %s", publish_id, status)

            if status == "PUBLISH_COMPLETE":
                post_ids = data.get("publicaly_available_post_id", [])
                video_id = post_ids[0] if post_ids else None
                url = f"https://www.tiktok.com/video/{video_id}" if video_id else None
                return "PUBLISH_COMPLETE", url, None

            if status in ("FAILED", "PUBLISH_FAILED"):
                fail_reason = data.get("fail_reason", "Unknown error")
                return "FAILED", None, f"TikTok publish failed: {fail_reason}"

        return (
            "PENDING",
            None,
            (
                f"TikTok is still processing after {self._polling_max_retries} attempts "
                f"(~{self._polling_max_retries * self._polling_interval / 60:.0f} min). "
                + (
                    "Check the TikTok app to confirm whether the post was completed."
                    if self._post_mode == _POST_MODE_DIRECT
                    else "Check your TikTok inbox — the draft should be there."
                )
            ),
        )

    def upload(
        self,
        video_path: Path,
        metadata: ClipMetadata,
        clip_index: int = 0,
    ) -> UploadResult:
        if self._access_token is None:
            self.authenticate()

        meta = apply_platform_limits(metadata, _PLATFORM)
        if self._post_mode == _POST_MODE_DIRECT:
            logger.info(
                "TikTok direct-post metadata prepared: title_len=%d description_len=%d hashtags=%d caption_len=%d privacy=%s",
                len(meta.title),
                len(meta.description),
                len(meta.hashtags),
                len(meta.caption),
                self._privacy_level,
            )
        else:
            logger.warning(
                "TikTok draft upload: the video is sent to the creator inbox for in-app editing. "
                "Caption and hashtags may need to be finalized inside TikTok before publishing."
            )
            logger.info(
                "TikTok draft metadata prepared: title_len=%d description_len=%d hashtags=%d caption_len=%d",
                len(meta.title),
                len(meta.description),
                len(meta.hashtags),
                len(meta.caption),
            )

        publish_id, upload_url, error = self._init_post(video_path, meta)
        if error:
            logger.error(error)
            return UploadResult(
                platform=_PLATFORM, clip_index=clip_index, status="failed", error=error
            )

        error = self._upload_chunks(video_path, upload_url)
        if error:
            logger.error(error)
            return UploadResult(
                platform=_PLATFORM, clip_index=clip_index, status="failed", error=error
            )

        if self._post_mode == _POST_MODE_DIRECT:
            logger.info("TikTok chunks uploaded for publish_id=%s. Polling publication status.", publish_id)
            status, url, error = self._poll_status(publish_id)
            if status == "PUBLISH_COMPLETE":
                return UploadResult(
                    platform=_PLATFORM,
                    clip_index=clip_index,
                    status="success",
                    url=url,
                )
            return UploadResult(
                platform=_PLATFORM,
                clip_index=clip_index,
                status="pending" if status == "PENDING" else "failed",
                url=url,
                error=error,
            )

        logger.info(
            "TikTok chunks uploaded for publish_id=%s. "
            "Video sent to TikTok draft flow — open the app inbox to review caption/hashtags and publish.",
            publish_id,
        )
        return UploadResult(
            platform=_PLATFORM,
            clip_index=clip_index,
            status="pending",
            error=(
                "Vídeo enviado para o rascunho do TikTok. Abra a caixa de entrada do app para "
                "revisar legenda/hashtags e concluir a publicação."
            ),
        )
