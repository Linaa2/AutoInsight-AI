"""Tests for utils.logger — ModuleLogger factory."""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest

from utils.logger import ModuleLogger, _registry, get_module_logger

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_test_loggers() -> None:  # type: ignore[return]
    """Remove all test loggers from both the registry and stdlib after each test."""
    yield
    for name in [k for k in list(_registry) if k.startswith("test.")]:
        underlying = logging.getLogger(name)
        for handler in underlying.handlers[:]:
            handler.close()
            underlying.removeHandler(handler)
        _registry.pop(name, None)


# ---------------------------------------------------------------------------
# Factory / caching behaviour
# ---------------------------------------------------------------------------


class TestGetModuleLogger:
    def test_returns_module_logger_instance(self, tmp_path: Path) -> None:
        result = get_module_logger("test.factory", log_dir=tmp_path, console=False)
        assert isinstance(result, ModuleLogger)

    def test_same_name_returns_cached_instance(self, tmp_path: Path) -> None:
        a = get_module_logger("test.cached", log_dir=tmp_path, console=False)
        b = get_module_logger("test.cached", log_dir=tmp_path, console=False)
        assert a is b

    def test_different_names_return_different_instances(self, tmp_path: Path) -> None:
        a = get_module_logger("test.mod_a", log_dir=tmp_path, console=False)
        b = get_module_logger("test.mod_b", log_dir=tmp_path, console=False)
        assert a is not b

    def test_name_property_matches_input(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.name_prop", log_dir=tmp_path, console=False)
        assert logger.name == "test.name_prop"

    def test_default_log_dir_used_when_none(self) -> None:
        """get_module_logger falls back to logs/ when log_dir is omitted."""
        # Patch DEFAULT_LOG_DIR to a tmp path so no real files escape the test
        import utils.logger as logger_mod

        original = logger_mod.DEFAULT_LOG_DIR
        logger_mod.DEFAULT_LOG_DIR = Path("/tmp/__autoinsight_test_default")
        try:
            result = get_module_logger("test.default_dir", console=False)
            assert isinstance(result, ModuleLogger)
        finally:
            # Cleanup the created logger + file
            underlying = logging.getLogger("test.default_dir")
            for h in underlying.handlers[:]:
                h.close()
                underlying.removeHandler(h)
            _registry.pop("test.default_dir", None)
            logger_mod.DEFAULT_LOG_DIR = original
            import shutil

            shutil.rmtree("/tmp/__autoinsight_test_default", ignore_errors=True)


# ---------------------------------------------------------------------------
# File creation and naming
# ---------------------------------------------------------------------------


class TestFileCreation:
    def test_log_file_is_created_on_first_call(self, tmp_path: Path) -> None:
        get_module_logger("test.filecreate", log_dir=tmp_path, console=False)
        files = list(tmp_path.glob("*.log"))
        assert len(files) == 1

    def test_filename_follows_name_date_pattern(self, tmp_path: Path) -> None:
        get_module_logger("test.filename", log_dir=tmp_path, console=False)
        # Pattern: test_filename_YYYYMMDD.log  (8-digit date)
        matches = list(tmp_path.glob("test_filename_????????.log"))
        assert len(matches) == 1

    def test_dots_in_name_become_underscores(self, tmp_path: Path) -> None:
        get_module_logger("my.dotted.module", log_dir=tmp_path, console=False)
        matches = list(tmp_path.glob("my_dotted_module_*.log"))
        assert len(matches) == 1

    def test_dashes_in_name_become_underscores(self, tmp_path: Path) -> None:
        get_module_logger("test.dash-name", log_dir=tmp_path, console=False)
        matches = list(tmp_path.glob("test_dash_name_*.log"))
        assert len(matches) == 1

    def test_different_modules_produce_separate_files(self, tmp_path: Path) -> None:
        get_module_logger("test.svc_a", log_dir=tmp_path, console=False)
        get_module_logger("test.svc_b", log_dir=tmp_path, console=False)
        assert len(list(tmp_path.glob("*.log"))) == 2

    def test_creates_log_dir_when_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "logs"
        assert not nested.exists()
        get_module_logger("test.nested_dir", log_dir=nested, console=False)
        assert nested.is_dir()
        assert len(list(nested.glob("*.log"))) == 1


# ---------------------------------------------------------------------------
# Log levels
# ---------------------------------------------------------------------------


class TestLogLevels:
    def _read_log(self, tmp_path: Path, safe_name: str) -> str:
        log_file = next(tmp_path.glob(f"{safe_name}_*.log"))
        return log_file.read_text(encoding="utf-8")

    def test_debug_written(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.lvl_debug", log_dir=tmp_path, console=False)
        logger.debug("debug payload")
        assert "debug payload" in self._read_log(tmp_path, "test_lvl_debug")

    def test_info_written(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.lvl_info", log_dir=tmp_path, console=False)
        logger.info("info payload")
        assert "info payload" in self._read_log(tmp_path, "test_lvl_info")

    def test_warning_written(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.lvl_warning", log_dir=tmp_path, console=False)
        logger.warning("warning payload")
        assert "warning payload" in self._read_log(tmp_path, "test_lvl_warning")

    def test_error_written(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.lvl_error", log_dir=tmp_path, console=False)
        logger.error("error payload")
        assert "error payload" in self._read_log(tmp_path, "test_lvl_error")

    def test_critical_written(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.lvl_critical", log_dir=tmp_path, console=False)
        logger.critical("critical payload")
        assert "critical payload" in self._read_log(tmp_path, "test_lvl_critical")

    def test_below_minimum_level_filtered_out(self, tmp_path: Path) -> None:
        logger = get_module_logger(
            "test.lvl_filter",
            level=logging.WARNING,
            log_dir=tmp_path,
            console=False,
        )
        logger.debug("below min — filtered")
        logger.info("below min — filtered")
        logger.warning("at min — kept")
        content = self._read_log(tmp_path, "test_lvl_filter")
        assert "filtered" not in content
        assert "at min — kept" in content


# ---------------------------------------------------------------------------
# Log record format
# ---------------------------------------------------------------------------


class TestLogFormat:
    def _first_line(self, tmp_path: Path, safe_name: str) -> str:
        log_file = next(tmp_path.glob(f"{safe_name}_*.log"))
        return log_file.read_text(encoding="utf-8").strip().splitlines()[0]

    def test_four_pipe_separated_fields(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.fmt_pipes", log_dir=tmp_path, console=False)
        logger.info("pipe check")
        line = self._first_line(tmp_path, "test_fmt_pipes")
        assert len(line.split(" | ")) == 4

    def test_logger_name_in_record(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.fmt_name", log_dir=tmp_path, console=False)
        logger.info("name check")
        line = self._first_line(tmp_path, "test_fmt_name")
        assert "test.fmt_name" in line

    def test_level_name_in_record(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.fmt_level", log_dir=tmp_path, console=False)
        logger.error("level check")
        line = self._first_line(tmp_path, "test_fmt_level")
        assert "ERROR" in line

    def test_message_in_record(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.fmt_msg", log_dir=tmp_path, console=False)
        logger.info("unique-msg-abc123")
        line = self._first_line(tmp_path, "test_fmt_msg")
        assert "unique-msg-abc123" in line

    def test_timestamp_starts_record(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.fmt_ts", log_dir=tmp_path, console=False)
        logger.info("ts check")
        line = self._first_line(tmp_path, "test_fmt_ts")
        # asctime format is YYYY-MM-DD HH:MM:SS,mmm
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)


# ---------------------------------------------------------------------------
# Rotation configuration
# ---------------------------------------------------------------------------


class TestRotation:
    def test_file_handler_is_timed_rotating(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.rot_type", log_dir=tmp_path, console=False)
        handlers = [h for h in logger._logger.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(handlers) == 1

    def test_rotation_interval_is_midnight(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.rot_when", log_dir=tmp_path, console=False)
        handler = next(
            h for h in logger._logger.handlers if isinstance(h, TimedRotatingFileHandler)
        )
        assert handler.when == "MIDNIGHT"

    def test_backup_count_is_30(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.rot_backup", log_dir=tmp_path, console=False)
        handler = next(
            h for h in logger._logger.handlers if isinstance(h, TimedRotatingFileHandler)
        )
        assert handler.backupCount == 30

    def test_rotation_suffix_format(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.rot_suffix", log_dir=tmp_path, console=False)
        handler = next(
            h for h in logger._logger.handlers if isinstance(h, TimedRotatingFileHandler)
        )
        assert handler.suffix == "%Y%m%d"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_returns_self_on_enter(self, tmp_path: Path) -> None:
        ml = get_module_logger("test.ctx_enter", log_dir=tmp_path, console=False)
        with ml as ctx:
            assert ctx is ml

    def test_can_log_inside_context(self, tmp_path: Path) -> None:
        with get_module_logger("test.ctx_log", log_dir=tmp_path, console=False) as logger:
            logger.info("inside context")
        log_file = next(tmp_path.glob("test_ctx_log_*.log"))
        assert "inside context" in log_file.read_text(encoding="utf-8")

    def test_handlers_closed_on_exit(self, tmp_path: Path) -> None:
        ml = get_module_logger("test.ctx_close", log_dir=tmp_path, console=False)
        underlying = ml._logger
        with ml:
            assert len(underlying.handlers) > 0
        assert len(underlying.handlers) == 0

    def test_registry_entry_removed_on_exit(self, tmp_path: Path) -> None:
        get_module_logger("test.ctx_reg", log_dir=tmp_path, console=False)
        assert "test.ctx_reg" in _registry
        with _registry["test.ctx_reg"]:
            pass
        assert "test.ctx_reg" not in _registry

    def test_new_logger_created_after_context_exit(self, tmp_path: Path) -> None:
        ml1 = get_module_logger("test.ctx_renew", log_dir=tmp_path, console=False)
        with ml1:
            pass
        ml2 = get_module_logger("test.ctx_renew", log_dir=tmp_path, console=False)
        assert ml1 is not ml2  # fresh instance after cleanup


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


class TestConsoleOutput:
    def test_console_stream_handler_present_when_enabled(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.con_on", log_dir=tmp_path, console=True)
        pure_stream = [h for h in logger._logger.handlers if type(h) is logging.StreamHandler]
        assert len(pure_stream) == 1

    def test_no_console_handler_when_disabled(self, tmp_path: Path) -> None:
        logger = get_module_logger("test.con_off", log_dir=tmp_path, console=False)
        pure_stream = [h for h in logger._logger.handlers if type(h) is logging.StreamHandler]
        assert len(pure_stream) == 0

    def test_console_handler_writes_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        logger = get_module_logger("test.con_stdout", log_dir=tmp_path, console=True)
        logger.info("stdout message")
        captured = capsys.readouterr()
        assert "stdout message" in captured.out


# ---------------------------------------------------------------------------
# caplog integration
# ---------------------------------------------------------------------------


class TestCaplog:
    def test_caplog_captures_info_records(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = get_module_logger("test.caplog_info", log_dir=tmp_path, console=False)
        # Manually attach caplog's handler since propagate=False
        logger._logger.addHandler(caplog.handler)
        with caplog.at_level(logging.INFO, logger="test.caplog_info"):
            logger.info("caplog info message")
        logger._logger.removeHandler(caplog.handler)
        assert "caplog info message" in caplog.text
        assert "INFO" in caplog.text

    def test_caplog_captures_warning_records(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = get_module_logger("test.caplog_warn", log_dir=tmp_path, console=False)
        logger._logger.addHandler(caplog.handler)
        with caplog.at_level(logging.WARNING, logger="test.caplog_warn"):
            logger.warning("caplog warning message")
        logger._logger.removeHandler(caplog.handler)
        assert "caplog warning message" in caplog.text
        assert "WARNING" in caplog.text

    def test_caplog_respects_level_filter(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = get_module_logger("test.caplog_filter", log_dir=tmp_path, console=False)
        logger._logger.addHandler(caplog.handler)
        with caplog.at_level(logging.WARNING, logger="test.caplog_filter"):
            logger.debug("should not appear")
            logger.info("should not appear either")
            logger.warning("should appear")
        logger._logger.removeHandler(caplog.handler)
        assert "should not appear" not in caplog.text
        assert "should appear" in caplog.text
