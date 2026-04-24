import json
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_PLATFORMS = frozenset({"youtube", "instagram", "tiktok"})


def _credentials_dir(token_dir: Path) -> Path:
    token_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    current_mode = stat.S_IMODE(os.stat(token_dir).st_mode)
    if current_mode != 0o700:
        os.chmod(token_dir, 0o700)
    return token_dir


def _token_path(platform: str, token_dir: Path) -> Path:
    if platform not in _SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported platform: {platform!r}. Must be one of {sorted(_SUPPORTED_PLATFORMS)}"
        )
    return token_dir / f"{platform}.json"


def get_token(platform: str, token_dir: Path) -> dict | None:
    path = _token_path(platform, token_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Token file for %r is corrupted; treating as absent: %s", platform, path)
        return None


def save_token(platform: str, token: dict, token_dir: Path) -> None:
    _credentials_dir(token_dir)
    path = _token_path(platform, token_dir)
    content = json.dumps(token, indent=2).encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)


def revoke_token(platform: str, token_dir: Path) -> None:
    path = _token_path(platform, token_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
