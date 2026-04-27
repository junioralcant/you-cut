from pathlib import Path

import pytest

from youcut.yt_dlp_auth import (
    YtDlpAuthConfig,
    YtDlpAuthConfigError,
    apply_yt_dlp_auth,
    resolve_yt_dlp_auth_config,
)


def test_resolve_yt_dlp_auth_config_prefers_browser():
    result = resolve_yt_dlp_auth_config({"YOUCUT_COOKIES_FROM_BROWSER": "chrome"})

    assert result == YtDlpAuthConfig(browser="chrome")


def test_resolve_yt_dlp_auth_config_uses_cookie_file(tmp_path):
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("data", encoding="utf-8")

    result = resolve_yt_dlp_auth_config({"YOUCUT_COOKIES_FILE": str(cookie_file)})

    assert result == YtDlpAuthConfig(cookie_file=cookie_file)


def test_resolve_yt_dlp_auth_config_uses_js_runtimes_only():
    result = resolve_yt_dlp_auth_config({"YOUCUT_YTDLP_JS_RUNTIMES": "node,quickjs"})

    assert result == YtDlpAuthConfig(js_runtimes=("node", "quickjs"))


def test_resolve_yt_dlp_auth_config_rejects_conflicting_env(tmp_path):
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("data", encoding="utf-8")

    with pytest.raises(YtDlpAuthConfigError, match="Defina apenas uma opção"):
        resolve_yt_dlp_auth_config(
            {
                "YOUCUT_COOKIES_FROM_BROWSER": "chrome",
                "YOUCUT_COOKIES_FILE": str(cookie_file),
            }
        )


def test_resolve_yt_dlp_auth_config_rejects_missing_cookie_file():
    with pytest.raises(YtDlpAuthConfigError, match="não existe"):
        resolve_yt_dlp_auth_config({"YOUCUT_COOKIES_FILE": "/tmp/does-not-exist.txt"})


def test_resolve_yt_dlp_auth_config_rejects_invalid_js_runtime():
    with pytest.raises(YtDlpAuthConfigError, match="Valores inválidos em YOUCUT_YTDLP_JS_RUNTIMES"):
        resolve_yt_dlp_auth_config({"YOUCUT_YTDLP_JS_RUNTIMES": "node,foo"})


def test_apply_yt_dlp_auth_sets_browser_option():
    ydl_opts = {"quiet": True}

    result = apply_yt_dlp_auth(ydl_opts, YtDlpAuthConfig(browser="chrome"))

    assert result["cookiesfrombrowser"] == ("chrome",)


def test_apply_yt_dlp_auth_sets_cookie_file_option(tmp_path):
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("data", encoding="utf-8")
    ydl_opts = {"quiet": True}

    result = apply_yt_dlp_auth(ydl_opts, YtDlpAuthConfig(cookie_file=Path(cookie_file)))

    assert result["cookiefile"] == str(cookie_file)


def test_apply_yt_dlp_auth_sets_js_runtimes():
    ydl_opts = {"quiet": True}

    result = apply_yt_dlp_auth(ydl_opts, YtDlpAuthConfig(js_runtimes=("node",)))

    assert result["js_runtimes"] == {"node": {}}
