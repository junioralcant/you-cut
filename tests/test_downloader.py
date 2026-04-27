from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp
from yt_dlp.utils import DownloadError

from youcut.downloader import VideoDownloadError, download_video
from youcut.yt_dlp_auth import YtDlpAuthConfig


FIXTURE_MP4 = Path(__file__).parent / "fixtures" / "sample.mp4"


class TestLocalFile:
    def test_existing_local_file_returned_directly(self):
        result = download_video(str(FIXTURE_MP4), Path("output"))
        assert result == FIXTURE_MP4

    def test_local_file_does_not_call_yt_dlp(self):
        with patch("youcut.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            download_video(str(FIXTURE_MP4), Path("output"))
            mock_ydl.assert_not_called()

    def test_missing_local_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado: /tmp/nonexistent.mp4"):
            download_video("/tmp/nonexistent.mp4", Path("output"))


class TestYouTubeDownload:
    def _make_ydl_context(self, output_dir: Path, filename: str):
        mock_info = {"title": "Test Video", "ext": "mp4"}
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl_instance.prepare_filename.return_value = str(output_dir / filename)
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        return mock_ydl_instance

    def test_uses_best_format(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance) as mock_cls:
            download_video("https://youtube.com/watch?v=abc", tmp_path)
            call_opts = mock_cls.call_args[0][0]
            assert call_opts["format"] == "bestvideo+bestaudio/best"

    def test_output_path_is_inside_output_dir(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance) as mock_cls:
            download_video("https://youtube.com/watch?v=abc", tmp_path)
            call_opts = mock_cls.call_args[0][0]
            assert call_opts["outtmpl"].startswith(str(tmp_path))

    def test_quiet_mode_enabled(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance) as mock_cls:
            download_video("https://youtube.com/watch?v=abc", tmp_path)
            call_opts = mock_cls.call_args[0][0]
            assert call_opts["quiet"] is True

    def test_returns_path_from_prepare_filename(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance):
            result = download_video("https://youtube.com/watch?v=abc", tmp_path)
            assert result == tmp_path / "video.mp4"

    def test_download_error_is_wrapped_with_friendly_message(self, tmp_path):
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = DownloadError("Video unavailable")
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)

        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance):
            with pytest.raises(VideoDownloadError, match="Falha ao baixar o vídeo"):
                download_video("https://youtube.com/watch?v=invalid", tmp_path)

    def test_http_url_triggers_yt_dlp(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance) as mock_cls:
            download_video("http://youtube.com/watch?v=abc", tmp_path)
            mock_cls.assert_called_once()

    def test_output_dir_is_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "new_output"
        assert not new_dir.exists()
        mock_instance = self._make_ydl_context(new_dir, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance):
            download_video("https://youtube.com/watch?v=abc", new_dir)
        assert new_dir.exists()

    def test_normalizes_shell_escaped_url_before_download(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance):
            download_video(r"https://www.youtube.com/watch\?v\=abc", tmp_path)

        mock_instance.extract_info.assert_called_once_with("https://www.youtube.com/watch?v=abc", download=True)

    def test_passes_cookie_file_to_yt_dlp(self, tmp_path):
        mock_instance = self._make_ydl_context(tmp_path, "video.mp4")
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("cookies", encoding="utf-8")

        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance) as mock_cls:
            download_video(
                "https://youtube.com/watch?v=abc",
                tmp_path,
                auth_config=YtDlpAuthConfig(cookie_file=cookie_file),
            )

        call_opts = mock_cls.call_args[0][0]
        assert call_opts["cookiefile"] == str(cookie_file)

    def test_download_error_appends_cookie_hint(self, tmp_path):
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = DownloadError("Sign in to confirm you're not a bot")
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)

        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_instance):
            with pytest.raises(VideoDownloadError, match="YOUCUT_COOKIES_FROM_BROWSER=chrome"):
                download_video("https://youtube.com/watch?v=invalid", tmp_path)
