import os
import stat
from pathlib import Path

import pytest

from youcut.uploader.auth import get_token, revoke_token, save_token


@pytest.fixture()
def token_dir(tmp_path: Path) -> Path:
    return tmp_path / "credentials"


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestSaveToken:
    def test_creates_file_with_permission_600(self, token_dir: Path) -> None:
        save_token("youtube", {"access_token": "abc"}, token_dir)
        token_file = token_dir / "youtube.json"
        assert token_file.exists()
        assert _file_mode(token_file) == 0o600

    def test_creates_credentials_dir_automatically(self, token_dir: Path) -> None:
        assert not token_dir.exists()
        save_token("youtube", {"access_token": "abc"}, token_dir)
        assert token_dir.exists()

    def test_credentials_dir_has_permission_700(self, token_dir: Path) -> None:
        save_token("youtube", {"access_token": "abc"}, token_dir)
        assert _file_mode(token_dir) == 0o700

    def test_overwrites_existing_token(self, token_dir: Path) -> None:
        save_token("youtube", {"access_token": "old"}, token_dir)
        save_token("youtube", {"access_token": "new"}, token_dir)
        assert get_token("youtube", token_dir) == {"access_token": "new"}

    def test_file_permission_600_after_overwrite(self, token_dir: Path) -> None:
        save_token("youtube", {"access_token": "old"}, token_dir)
        save_token("youtube", {"access_token": "new"}, token_dir)
        assert _file_mode(token_dir / "youtube.json") == 0o600


class TestGetToken:
    def test_returns_saved_token(self, token_dir: Path) -> None:
        token = {"access_token": "xyz", "refresh_token": "zzz"}
        save_token("instagram", token, token_dir)
        assert get_token("instagram", token_dir) == token

    def test_returns_none_for_missing_token(self, token_dir: Path) -> None:
        token_dir.mkdir(parents=True, exist_ok=True)
        assert get_token("tiktok", token_dir) is None

    def test_returns_none_when_dir_does_not_exist(self, token_dir: Path) -> None:
        assert get_token("youtube", token_dir) is None

    def test_isolates_platforms(self, token_dir: Path) -> None:
        save_token("youtube", {"token": "yt"}, token_dir)
        save_token("instagram", {"token": "ig"}, token_dir)
        assert get_token("youtube", token_dir) == {"token": "yt"}
        assert get_token("instagram", token_dir) == {"token": "ig"}
        assert get_token("tiktok", token_dir) is None


class TestRevokeToken:
    def test_removes_token_file(self, token_dir: Path) -> None:
        save_token("youtube", {"access_token": "abc"}, token_dir)
        revoke_token("youtube", token_dir)
        assert not (token_dir / "youtube.json").exists()
        assert get_token("youtube", token_dir) is None

    def test_no_exception_when_file_does_not_exist(self, token_dir: Path) -> None:
        token_dir.mkdir(parents=True, exist_ok=True)
        revoke_token("youtube", token_dir)  # must not raise

    def test_no_exception_when_dir_does_not_exist(self, token_dir: Path) -> None:
        revoke_token("youtube", token_dir)  # must not raise

    def test_revoke_does_not_affect_other_platforms(self, token_dir: Path) -> None:
        save_token("youtube", {"token": "yt"}, token_dir)
        save_token("instagram", {"token": "ig"}, token_dir)
        revoke_token("youtube", token_dir)
        assert get_token("youtube", token_dir) is None
        assert get_token("instagram", token_dir) == {"token": "ig"}


class TestEdgeCases:
    def test_get_token_returns_none_for_corrupted_file(self, token_dir: Path) -> None:
        token_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        corrupt_file = token_dir / "youtube.json"
        corrupt_file.write_text("not-valid-json", encoding="utf-8")
        assert get_token("youtube", token_dir) is None

    def test_unsupported_platform_raises_value_error(self, token_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported platform"):
            save_token("twitter", {"token": "x"}, token_dir)

    def test_unsupported_platform_get_raises_value_error(self, token_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported platform"):
            get_token("twitter", token_dir)

    def test_file_created_with_correct_permissions_atomically(self, token_dir: Path) -> None:
        save_token("tiktok", {"access_token": "tok"}, token_dir)
        assert _file_mode(token_dir / "tiktok.json") == 0o600
