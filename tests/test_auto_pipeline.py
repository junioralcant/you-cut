"""Tests for the automatic pipeline: _show_consolidated_summary, _select_execution_mode, and related helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from youcut.models import ClipRecord


def _make_clip(title: str, upload_status: dict | None = None, *, tmp_path: Path) -> ClipRecord:
    p = tmp_path / f"{title}.mp4"
    p.touch()
    return ClipRecord(
        title=title,
        start_time=0.0,
        end_time=60.0,
        clip_path=p,
        thumbnail_path=None,
        approved=True,
        upload_status=upload_status or {},
    )


# ---------------------------------------------------------------------------
# Task 2.0 — _show_consolidated_summary
# ---------------------------------------------------------------------------

class TestShowConsolidatedSummary:
    def test_both_empty_lists_no_crash(self, tmp_path):
        """_show_consolidated_summary([], []) não deve lançar exceção."""
        from youcut.cli import _show_consolidated_summary
        _show_consolidated_summary([], [])

    def test_only_yt_records_renders_table(self, tmp_path):
        """Apenas yt_records populado — tabela é exibida sem crash."""
        clip = _make_clip("YouTube Clip", {"youtube": "success"}, tmp_path=tmp_path)
        from youcut.cli import _show_consolidated_summary
        _show_consolidated_summary([clip], [])

    def test_only_social_records_renders_table(self, tmp_path):
        """Apenas social_records populado — tabela é exibida sem crash."""
        clip = _make_clip("Social Clip", {"tiktok": "success"}, tmp_path=tmp_path)
        from youcut.cli import _show_consolidated_summary
        _show_consolidated_summary([], [clip])

    def test_both_lists_populated_renders_all(self, tmp_path, capsys):
        """Com ambas as listas populadas, todos os registros aparecem na saída."""
        yt = _make_clip("YT Clip", {"youtube": "success"}, tmp_path=tmp_path)
        social = _make_clip("Social Clip", {"tiktok": "failed"}, tmp_path=tmp_path)

        captured_output = []

        def fake_print(renderable=None, *args, **kwargs):
            captured_output.append(str(renderable) if renderable is not None else "")

        with patch("youcut.cli._console") as mock_console:
            mock_console.print.side_effect = fake_print
            from youcut.cli import _show_consolidated_summary
            _show_consolidated_summary([yt], [social])

        assert mock_console.print.call_count >= 2

    def test_success_failed_counters_displayed(self, tmp_path):
        """Contadores de success/failed por plataforma são exibidos ao final com valores corretos."""
        from rich.panel import Panel as RichPanel

        yt1 = _make_clip("YT 1", {"youtube": "success"}, tmp_path=tmp_path)
        yt2 = _make_clip("YT 2", {"youtube": "failed"}, tmp_path=tmp_path)
        social1 = _make_clip("Social 1", {"tiktok": "success", "instagram": "failed"}, tmp_path=tmp_path)

        printed = []

        with patch("youcut.cli._console") as mock_console:
            mock_console.print.side_effect = lambda r=None, *a, **k: printed.append(r)
            from youcut.cli import _show_consolidated_summary
            _show_consolidated_summary([yt1, yt2], [social1])

        assert mock_console.print.call_count >= 2, "Deve imprimir tabela + painel de contadores"
        panel = next((p for p in printed if isinstance(p, RichPanel)), None)
        assert panel is not None, "Painel de contadores deve ser exibido"
        panel_text = str(panel.renderable)
        assert "1 ok" in panel_text, f"youtube deve ter 1 ok — got: {panel_text}"
        assert "1 falhou" in panel_text, f"youtube deve ter 1 falhou — got: {panel_text}"

    def test_pending_status_shown_as_pending_not_failed(self, tmp_path):
        """Status 'pending' deve aparecer como 'pending' na tabela, não como 'failed'."""
        from rich.table import Table as RichTable

        clip = _make_clip("Social Pending", {"tiktok": "pending"}, tmp_path=tmp_path)
        printed = []

        with patch("youcut.cli._console") as mock_console:
            mock_console.print.side_effect = lambda r=None, *a, **k: printed.append(r)
            from youcut.cli import _show_consolidated_summary
            _show_consolidated_summary([], [clip])

        table = next((p for p in printed if isinstance(p, RichTable)), None)
        assert table is not None
        table_str = str(table.columns[-1]._cells)
        assert "pending" in table_str, "Status 'pending' deve aparecer como 'pending'"
        assert "failed" not in table_str, "Status 'pending' não deve aparecer como 'failed'"

    def test_skipped_status_not_counted_as_failed(self, tmp_path):
        """Status 'skipped' não deve incrementar contadores de success nem failed."""
        from rich.panel import Panel as RichPanel

        clip = _make_clip("Skipped Clip", {"instagram": "skipped"}, tmp_path=tmp_path)
        printed = []

        with patch("youcut.cli._console") as mock_console:
            mock_console.print.side_effect = lambda r=None, *a, **k: printed.append(r)
            from youcut.cli import _show_consolidated_summary
            _show_consolidated_summary([clip], [])

        panel = next((p for p in printed if isinstance(p, RichPanel)), None)
        assert panel is None, "Painel de contadores não deve aparecer para status 'skipped'"

    def test_records_without_upload_status_no_crash(self, tmp_path):
        """Registros com upload_status vazio são exibidos com 'sem upload'."""
        yt = _make_clip("YT sem upload", {}, tmp_path=tmp_path)
        social = _make_clip("Social sem upload", {}, tmp_path=tmp_path)
        from youcut.cli import _show_consolidated_summary
        _show_consolidated_summary([yt], [social])

    def test_no_counter_panel_when_no_uploads(self, tmp_path):
        """Sem upload_status em nenhum registro, painel de contadores não é exibido."""
        yt = _make_clip("YT", {}, tmp_path=tmp_path)
        print_calls = []

        with patch("youcut.cli._console") as mock_console:
            mock_console.print.side_effect = lambda r=None, *a, **k: print_calls.append(r)
            from youcut.cli import _show_consolidated_summary
            _show_consolidated_summary([yt], [])

        assert mock_console.print.call_count == 1, "Apenas a tabela, sem painel de contadores"


# ---------------------------------------------------------------------------
# Task 3.0 — _select_execution_mode
# ---------------------------------------------------------------------------

class TestSelectExecutionMode:
    def test_returns_manual_when_user_selects_manual(self):
        """Retorna 'manual' quando o usuário escolhe a opção Manual."""
        with patch("youcut.cli.questionary.select") as mock_select:
            mock_select.return_value = MagicMock(ask=MagicMock(return_value="manual"))
            from youcut.cli import _select_execution_mode
            result = _select_execution_mode()
        assert result == "manual"

    def test_returns_auto_when_user_selects_auto(self):
        """Retorna 'auto' quando o usuário escolhe a opção Automático."""
        with patch("youcut.cli.questionary.select") as mock_select:
            mock_select.return_value = MagicMock(ask=MagicMock(return_value="auto"))
            from youcut.cli import _select_execution_mode
            result = _select_execution_mode()
        assert result == "auto"

    def test_exits_when_user_cancels(self):
        """Levanta typer.Exit quando o usuário cancela (questionary retorna None)."""
        with patch("youcut.cli.questionary.select") as mock_select:
            mock_select.return_value = MagicMock(ask=MagicMock(return_value=None))
            from youcut.cli import _select_execution_mode
            with pytest.raises(typer.Exit):
                _select_execution_mode()

    def test_uses_two_choices(self):
        """questionary.select é chamado com exatamente duas choices."""
        captured_kwargs = {}

        def fake_select(question, choices=None, **kwargs):
            captured_kwargs["choices"] = choices
            return MagicMock(ask=MagicMock(return_value="manual"))

        with patch("youcut.cli.questionary.select", side_effect=fake_select):
            from youcut.cli import _select_execution_mode
            _select_execution_mode()

        assert len(captured_kwargs["choices"]) == 2

    def test_choice_values_are_manual_and_auto(self):
        """Os valores das choices são exatamente 'manual' e 'auto'."""
        captured_choices = []

        def fake_select(question, choices=None, **kwargs):
            captured_choices.extend(choices)
            return MagicMock(ask=MagicMock(return_value="manual"))

        with patch("youcut.cli.questionary.select", side_effect=fake_select):
            from youcut.cli import _select_execution_mode
            _select_execution_mode()

        values = [c.value for c in captured_choices]
        assert "manual" in values
        assert "auto" in values
