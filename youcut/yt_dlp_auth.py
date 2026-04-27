import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class YtDlpAuthConfigError(ValueError):
    pass


@dataclass(frozen=True)
class YtDlpAuthConfig:
    browser: str | None = None
    cookie_file: Path | None = None
    js_runtimes: tuple[str, ...] = ()


def resolve_yt_dlp_auth_config(env: Mapping[str, str] | None = None) -> YtDlpAuthConfig | None:
    source_env = env if env is not None else os.environ
    browser = (source_env.get("YOUCUT_COOKIES_FROM_BROWSER") or "").strip()
    cookie_file_raw = (source_env.get("YOUCUT_COOKIES_FILE") or "").strip()
    js_runtimes_raw = (source_env.get("YOUCUT_YTDLP_JS_RUNTIMES") or "").strip()
    js_runtimes = tuple(_parse_js_runtimes(js_runtimes_raw))

    if browser and cookie_file_raw:
        raise YtDlpAuthConfigError(
            "Defina apenas uma opção de autenticação do yt-dlp: "
            "YOUCUT_COOKIES_FROM_BROWSER ou YOUCUT_COOKIES_FILE."
        )

    if browser:
        return YtDlpAuthConfig(browser=browser, js_runtimes=js_runtimes)

    if cookie_file_raw:
        cookie_file = Path(cookie_file_raw).expanduser()
        if not cookie_file.exists():
            raise YtDlpAuthConfigError(
                f"Arquivo informado em YOUCUT_COOKIES_FILE não existe: {cookie_file}"
            )
        return YtDlpAuthConfig(cookie_file=cookie_file, js_runtimes=js_runtimes)

    if js_runtimes:
        return YtDlpAuthConfig(js_runtimes=js_runtimes)

    return None


def _parse_js_runtimes(raw: str) -> list[str]:
    if not raw:
        return []

    allowed = {"deno", "node", "bun", "quickjs"}
    runtimes: list[str] = []
    invalid: list[str] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in allowed:
            invalid.append(item)
            continue
        if item not in runtimes:
            runtimes.append(item)

    if invalid:
        valid_str = ", ".join(sorted(allowed))
        invalid_str = ", ".join(invalid)
        raise YtDlpAuthConfigError(
            f"Valores inválidos em YOUCUT_YTDLP_JS_RUNTIMES: {invalid_str}. "
            f"Use apenas: {valid_str}."
        )

    return runtimes


def apply_yt_dlp_auth(ydl_opts: dict, auth_config: YtDlpAuthConfig | None) -> dict:
    if auth_config is None:
        return ydl_opts

    if auth_config.js_runtimes:
        ydl_opts["js_runtimes"] = {runtime: {} for runtime in auth_config.js_runtimes}

    if auth_config.browser:
        ydl_opts["cookiesfrombrowser"] = (auth_config.browser,)
    elif auth_config.cookie_file:
        ydl_opts["cookiefile"] = str(auth_config.cookie_file)

    return ydl_opts


def append_yt_auth_hint(message: str) -> str:
    if "Sign in to confirm you’re not a bot" not in message and "Sign in to confirm you're not a bot" not in message:
        return message

    hint = (
        " Dica: configure YOUCUT_COOKIES_FROM_BROWSER=chrome no .env "
        "ou YOUCUT_COOKIES_FILE=/caminho/cookies.txt para acessar via sessão autenticada."
    )
    if hint.strip() in message:
        return message
    return f"{message}{hint}"
