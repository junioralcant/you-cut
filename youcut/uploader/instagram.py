import logging
import os
import secrets
import socket
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .auth import get_token, save_token
from .base import ClipMetadata, UploadResult, Uploader
from .metadata import apply_platform_limits

logger = logging.getLogger(__name__)

_PLATFORM = "instagram"
_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"
_FACEBOOK_OAUTH_BASE = "https://www.facebook.com/v21.0/dialog/oauth"
_RUPLOAD_BASE = "https://rupload.facebook.com/video-upload/v21.0"
_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB
_POLLING_INTERVAL = 10.0  # seconds between status checks
_POLLING_TIMEOUT = 300.0  # 5 minutes
_ELIGIBLE_ACCOUNT_TYPES = frozenset({"BUSINESS", "MEDIA_CREATOR"})
_AUTH_TIMEOUT = 300.0
_INSTAGRAM_SCOPES = (
    "instagram_basic",
    "instagram_content_publish",
    "pages_show_list",
    "business_management",
)


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server_version = "YouCutInstagramOAuth/1.0"

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


class InstagramUploader(Uploader):
    def __init__(
        self,
        token_dir: Path,
        transport: httpx.BaseTransport | None = None,
        polling_interval: float = _POLLING_INTERVAL,
        polling_timeout: float = _POLLING_TIMEOUT,
    ) -> None:
        super().__init__(token_dir)
        self._access_token: str | None = None
        self._ig_user_id: str | None = None
        self._transport = transport
        self._polling_interval = polling_interval
        self._polling_timeout = polling_timeout

    @property
    def platform_name(self) -> str:
        return _PLATFORM

    def _make_client(self) -> httpx.Client:
        kwargs: dict = {"timeout": 60.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _find_free_port(self) -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_callback_server(self) -> tuple[_OAuthHTTPServer, str]:
        port = self._find_free_port()
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

        raise RuntimeError("Instagram OAuth callback timed out.")

    def _build_authorization_url(self, app_id: str, redirect_uri: str, state: str) -> str:
        query = urlencode(
            {
                "client_id": app_id,
                "redirect_uri": redirect_uri,
                "scope": ",".join(_INSTAGRAM_SCOPES),
                "response_type": "code",
                "state": state,
            }
        )
        return f"{_FACEBOOK_OAUTH_BASE}?{query}"

    def _exchange_code_for_token(
        self,
        *,
        app_id: str,
        app_secret: str,
        redirect_uri: str,
        code: str,
    ) -> str:
        with self._make_client() as client:
            response = client.get(
                f"{_GRAPH_API_BASE}/oauth/access_token",
                params={
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "redirect_uri": redirect_uri,
                    "code": code,
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Instagram OAuth token exchange failed: HTTP {response.status_code}: {response.text}"
            )

        access_token = response.json().get("access_token")
        if not access_token:
            raise RuntimeError("Instagram OAuth token exchange returned no access_token.")
        return access_token

    def _resolve_ig_user_id(self, access_token: str) -> str:
        with self._make_client() as client:
            response = client.get(
                f"{_GRAPH_API_BASE}/me/accounts",
                params={
                    "fields": "instagram_business_account{id},name",
                    "access_token": access_token,
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Instagram account discovery failed: HTTP {response.status_code}: {response.text}"
            )

        pages = response.json().get("data", [])
        for page in pages:
            instagram_account = page.get("instagram_business_account") or {}
            ig_user_id = instagram_account.get("id")
            if ig_user_id:
                return str(ig_user_id)

        raise RuntimeError(
            "Instagram OAuth succeeded, but no Instagram Business account linked to a Facebook Page was found."
        )

    def _run_oauth_flow(self) -> tuple[str, str]:
        app_id = os.getenv("INSTAGRAM_APP_ID")
        app_secret = os.getenv("INSTAGRAM_APP_SECRET")
        if not app_id or not app_secret:
            raise RuntimeError(
                "Instagram OAuth requires INSTAGRAM_APP_ID and INSTAGRAM_APP_SECRET to be configured before login."
            )

        state = secrets.token_urlsafe(24)
        server, redirect_uri = self._start_callback_server()
        auth_url = self._build_authorization_url(app_id, redirect_uri, state)

        logger.info("Starting Instagram OAuth flow — browser will open.")
        browser_thread = Thread(target=webbrowser.open, args=(auth_url,), kwargs={"new": 1}, daemon=True)
        browser_thread.start()

        try:
            code, returned_state, error = self._wait_for_callback(server)
        finally:
            server.server_close()

        if error:
            raise RuntimeError(f"Instagram authorization failed: {error}")
        if returned_state != state:
            raise RuntimeError("Instagram OAuth state mismatch.")
        if not code:
            raise RuntimeError("Instagram OAuth callback returned no authorization code.")

        access_token = self._exchange_code_for_token(
            app_id=app_id,
            app_secret=app_secret,
            redirect_uri=redirect_uri,
            code=code,
        )
        ig_user_id = self._resolve_ig_user_id(access_token)
        return access_token, ig_user_id

    def authenticate(self) -> None:
        token = get_token(_PLATFORM, self.token_dir)
        if token is not None:
            logger.debug("Reusing saved Instagram credentials.")
            self._access_token = token["access_token"]
            self._ig_user_id = token["ig_user_id"]
            return

        env_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        env_user_id = os.getenv("INSTAGRAM_USER_ID")
        if env_token and env_user_id:
            self._access_token = env_token
            self._ig_user_id = env_user_id
            save_token(
                _PLATFORM,
                {"access_token": env_token, "ig_user_id": env_user_id},
                self.token_dir,
            )
            logger.info("Instagram credentials saved from environment variables.")
            return

        access_token, ig_user_id = self._run_oauth_flow()
        self._access_token = access_token
        self._ig_user_id = ig_user_id
        save_token(
            _PLATFORM,
            {"access_token": access_token, "ig_user_id": ig_user_id},
            self.token_dir,
        )
        logger.info("Instagram OAuth credentials saved.")

    def _verify_business_account(self) -> str | None:
        """Returns descriptive error if account is not eligible, None if eligible."""
        with self._make_client() as client:
            try:
                resp = client.get(
                    f"{_GRAPH_API_BASE}/{self._ig_user_id}",
                    params={"fields": "account_type", "access_token": self._access_token},
                )
            except httpx.HTTPError as exc:
                return f"Failed to verify account type: {exc}"

        if resp.status_code != 200:
            return f"Failed to verify account type: HTTP {resp.status_code}: {resp.text}"

        account_type = resp.json().get("account_type", "")
        if account_type not in _ELIGIBLE_ACCOUNT_TYPES:
            return (
                f"Instagram account is not eligible for Reels publishing. "
                f"Account type: {account_type!r}. "
                "Only BUSINESS or MEDIA_CREATOR accounts can publish Reels via the API."
            )
        return None

    def _create_container(self, caption: str) -> tuple[str | None, str | None]:
        """Create Reels media container. Returns (container_id, error_msg)."""
        with self._make_client() as client:
            try:
                resp = client.post(
                    f"{_GRAPH_API_BASE}/{self._ig_user_id}/media",
                    data={
                        "media_type": "REELS",
                        "caption": caption,
                        "access_token": self._access_token,
                    },
                )
            except httpx.HTTPError as exc:
                return None, f"Container creation failed: {exc}"

        if resp.status_code != 200:
            return None, f"Container creation failed: HTTP {resp.status_code}: {resp.text}"

        container_id = resp.json().get("id")
        if not container_id:
            return None, f"Container creation returned no ID: {resp.text}"

        logger.debug("Created Instagram Reels container: %s", container_id)
        return container_id, None

    def _upload_video(self, container_id: str, video_path: Path) -> str | None:
        """Upload video in chunks via rupload.facebook.com. Returns error or None."""
        file_size = video_path.stat().st_size
        offset = 0

        with video_path.open("rb") as f, self._make_client() as client:
            while offset < file_size:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break

                try:
                    resp = client.post(
                        f"{_RUPLOAD_BASE}/{container_id}",
                        headers={
                            "Authorization": f"OAuth {self._access_token}",
                            "offset": str(offset),
                            "file_size": str(file_size),
                            "Content-Type": "video/mp4",
                        },
                        content=chunk,
                    )
                except httpx.HTTPError as exc:
                    return f"Video upload failed at offset {offset}: {exc}"

                if resp.status_code not in (200, 204):
                    return f"Video upload failed: HTTP {resp.status_code}: {resp.text}"

                offset += len(chunk)
                logger.debug("Uploaded %d/%d bytes for container %s", offset, file_size, container_id)

        return None

    def _poll_status(self, container_id: str) -> tuple[str, str | None]:
        """Poll container status until FINISHED, ERROR, or timeout.

        Returns (final_status, error_msg).
        """
        deadline = time.monotonic() + self._polling_timeout

        while time.monotonic() < deadline:
            with self._make_client() as client:
                try:
                    resp = client.get(
                        f"{_GRAPH_API_BASE}/{container_id}",
                        params={"fields": "status_code", "access_token": self._access_token},
                    )
                except httpx.HTTPError as exc:
                    return "ERROR", f"Status polling failed: {exc}"

            if resp.status_code != 200:
                return "ERROR", f"Status polling failed: HTTP {resp.status_code}: {resp.text}"

            status_code = resp.json().get("status_code", "")
            logger.debug("Container %s status: %s", container_id, status_code)

            if status_code == "FINISHED":
                return "FINISHED", None
            if status_code == "ERROR":
                return "ERROR", f"Container processing error (container={container_id})"
            if status_code == "EXPIRED":
                return "ERROR", f"Container expired before publishing (container={container_id})"

            time.sleep(self._polling_interval)

        return "ERROR", f"Container status polling timed out after {self._polling_timeout:.0f}s"

    def _publish_container(self, container_id: str) -> tuple[str | None, str | None]:
        """Publish the container and return (media_id, error_msg)."""
        with self._make_client() as client:
            try:
                resp = client.post(
                    f"{_GRAPH_API_BASE}/{self._ig_user_id}/media_publish",
                    data={
                        "creation_id": container_id,
                        "access_token": self._access_token,
                    },
                )
            except httpx.HTTPError as exc:
                return None, f"Publishing failed: {exc}"

        if resp.status_code != 200:
            return None, f"Publishing failed: HTTP {resp.status_code}: {resp.text}"

        media_id = resp.json().get("id")
        if not media_id:
            return None, f"Publishing returned no media ID: {resp.text}"

        return media_id, None

    def upload(
        self,
        video_path: Path,
        metadata: ClipMetadata,
        clip_index: int = 0,
    ) -> UploadResult:
        if self._access_token is None:
            self.authenticate()

        error = self._verify_business_account()
        if error:
            logger.error(error)
            return UploadResult(platform=_PLATFORM, clip_index=clip_index, status="failed", error=error)

        meta = apply_platform_limits(metadata, _PLATFORM)

        container_id, error = self._create_container(meta.caption)
        if error:
            logger.error(error)
            return UploadResult(platform=_PLATFORM, clip_index=clip_index, status="failed", error=error)

        error = self._upload_video(container_id, video_path)
        if error:
            logger.error(error)
            return UploadResult(platform=_PLATFORM, clip_index=clip_index, status="failed", error=error)

        final_status, error = self._poll_status(container_id)
        if final_status != "FINISHED":
            logger.error("Container %s failed: %s", container_id, error)
            return UploadResult(platform=_PLATFORM, clip_index=clip_index, status="failed", error=error)

        media_id, error = self._publish_container(container_id)
        if error:
            logger.error(error)
            return UploadResult(platform=_PLATFORM, clip_index=clip_index, status="failed", error=error)

        url = f"https://www.instagram.com/reel/{media_id}/"
        logger.info("Instagram upload succeeded: %s", url)
        return UploadResult(platform=_PLATFORM, clip_index=clip_index, status="success", url=url)
