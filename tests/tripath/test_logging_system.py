import logging
from pathlib import Path

import pytest

from src.tripath.utils.logger import log_pipeline_flag, setup_logger
from src.tripath.utils.tracing import trace_execution, trace_pipeline_stage


def test_logger_setup(tmp_path: Path):
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_module", log_level="DEBUG", log_file=log_file)

    assert logger.name == "test_module"
    assert logger.level == logging.DEBUG

    logger.info("Test message for logger setup")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test message for logger setup" in content


def test_log_pipeline_flag(tmp_path: Path):
    log_file = tmp_path / "flag_test.log"
    logger = setup_logger("test_flag_logger", log_file=log_file)

    log_pipeline_flag("use_blip2", True, reason="High quality captioning", logger=logger)

    content = log_file.read_text(encoding="utf-8")
    assert "[PIPELINE_FLAG] use_blip2 = True (Reason: High quality captioning)" in content


def test_trace_execution_decorator(tmp_path: Path):
    log_file = tmp_path / "trace_test.log"
    logger = setup_logger("src.tripath.test_func", log_level="DEBUG", log_file=log_file)

    @trace_execution(level=logging.DEBUG, logger=logger, log_args=True)
    def sample_func(x: int, y: int) -> int:
        return x + y

    result = sample_func(10, 20)
    assert result == 30

    content = log_file.read_text(encoding="utf-8")
    assert "[TRACE START] Entering sample_func" in content
    assert "[TRACE SUCCESS] Exited sample_func in" in content


def test_trace_execution_decorator_exception(tmp_path: Path):
    log_file = tmp_path / "trace_exc.log"
    logger = setup_logger("src.tripath.test_exc", log_level="DEBUG", log_file=log_file)

    @trace_execution(level=logging.DEBUG, logger=logger)
    def failing_func():
        raise ValueError("Boom!")

    with pytest.raises(ValueError, match="Boom!"):
        failing_func()

    content = log_file.read_text(encoding="utf-8")
    assert "[TRACE FAILED] failing_func failed after" in content
    assert "ValueError" in content


def test_trace_pipeline_stage_context(tmp_path: Path):
    log_file = tmp_path / "stage_test.log"
    setup_logger("docureason.pipeline", log_level="INFO", log_file=log_file)

    with trace_pipeline_stage("Ingestion Test Stage", logger_name="docureason.pipeline"):
        _ = 1 + 1

    content = log_file.read_text(encoding="utf-8")
    assert ">>> STAGE START: Ingestion Test Stage" in content
    assert ">>> STAGE COMPLETED: Ingestion Test Stage in" in content
