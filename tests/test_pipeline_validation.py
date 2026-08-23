"""test_pipeline_validation.py — Unit & Integration Tests for Input Validation and Configuration Transparency."""

from __future__ import annotations

from pathlib import Path

import pytest

from docureason import (
    DocuReasonConfig,
    DocuReasonPipeline,
    InvalidCorpusError,
    MissingRequiredConfigError,
)


def test_missing_input_dir_raises_error(tmp_path: Path):
    """Verify that omitting input_dir raises MissingRequiredConfigError."""
    with pytest.raises(MissingRequiredConfigError) as exc_info:
        DocuReasonPipeline(input_dir="", output_dir=tmp_path / "artifacts")
    assert "input_dir" in str(exc_info.value)


def test_missing_output_dir_raises_error(tmp_path: Path):
    """Verify that omitting output_dir raises MissingRequiredConfigError."""
    sample_dir = tmp_path / "corpus"
    sample_dir.mkdir()
    (sample_dir / "sample.txt").write_text("Test content", encoding="utf-8")

    with pytest.raises(MissingRequiredConfigError) as exc_info:
        DocuReasonConfig.validate_inputs(input_dir=sample_dir, output_dir="")
    assert "output_dir" in str(exc_info.value)


def test_nonexistent_input_dir_raises_invalid_corpus(tmp_path: Path):
    """Verify that specifying a non-existent directory raises InvalidCorpusError."""
    fake_dir = tmp_path / "does_not_exist"
    with pytest.raises(InvalidCorpusError) as exc_info:
        DocuReasonPipeline(input_dir=fake_dir, output_dir=tmp_path / "out")
    assert "does not exist" in str(exc_info.value)


def test_input_dir_is_file_raises_invalid_corpus(tmp_path: Path):
    """Verify that passing a file instead of a directory raises InvalidCorpusError."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("Hello world", encoding="utf-8")

    with pytest.raises(InvalidCorpusError) as exc_info:
        DocuReasonPipeline(input_dir=file_path, output_dir=tmp_path / "out")
    assert "is a file, not a directory" in str(exc_info.value)


def test_empty_corpus_directory_raises_invalid_corpus(tmp_path: Path):
    """Verify that an input directory with no supported files raises InvalidCorpusError."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    (empty_dir / "image.exe").write_text("binary data", encoding="utf-8")

    with pytest.raises(InvalidCorpusError) as exc_info:
        DocuReasonPipeline(input_dir=empty_dir, output_dir=tmp_path / "out")
    assert "No supported document files found" in str(exc_info.value)


def test_valid_corpus_validation_and_transparency_summary(tmp_path: Path, capsys):
    """Verify successful validation and configuration transparency printout."""
    corpus_dir = tmp_path / "valid_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_text("Dummy PDF content", encoding="utf-8")
    (corpus_dir / "doc2.txt").write_text("Dummy TXT content", encoding="utf-8")
    out_dir = tmp_path / "output_artifacts"

    pipeline = DocuReasonPipeline(input_dir=corpus_dir, output_dir=out_dir, config="balanced", verbose=True)

    assert pipeline.input_dir.resolve() == corpus_dir.resolve()
    assert pipeline.output_dir.resolve() == out_dir.resolve()
    assert pipeline.config.preset == "balanced"

    summary_txt = pipeline.config.summary_string(input_dir=corpus_dir, output_dir=out_dir)

    assert "[DocuReason Framework v1.1.3] Configuration & Pipeline Transparency" in summary_txt
    assert "input_dir (Data Corpus Path)" in summary_txt
    assert "VERIFIED - 2 supported file(s)" in summary_txt
    assert "output_dir (Artifact Path)" in summary_txt
    assert "Config Preset Profile       : 'balanced'" in summary_txt


def test_verbose_false_silences_transparency_logs(tmp_path: Path):
    """Verify that setting verbose=False silences startup transparency logs."""
    corpus_dir = tmp_path / "silent_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.txt").write_text("Content", encoding="utf-8")
    out_dir = tmp_path / "out"

    config = DocuReasonConfig.load_preset("quality_max")
    pipeline = DocuReasonPipeline(input_dir=corpus_dir, output_dir=out_dir, config=config, verbose=False)

    assert pipeline.verbose is False
    assert pipeline.config.preset == "quality_max"
