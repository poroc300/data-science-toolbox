"""Unit tests for functions in paths.py."""

import json
import os

import pytest

from dscience_toolbox.data_io import list_files


@pytest.mark.parametrize(
    "extensions, nfiles", ((".txt", 1), ([".txt"], 1), ([".txt", ".json"], 2))
)
def test_list_files_input_extensions(tmp_path, extensions, nfiles):
    """Test list_files with different input types."""
    # create a temporary txt and json file
    file_txt = tmp_path / "test.txt"
    file_json = tmp_path / "test.json"
    file_txt.write_text("a\nb\nc\n", encoding="utf-8")
    file_json.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")

    # assertions
    assert len(list_files(tmp_path, extensions)) == nfiles


@pytest.mark.parametrize(
    "extensions, nfiles, is_recursive",
    (([".txt"], 0, False), ([".txt"], 1, True), ([".txt", ".json"], 2, True)),
)
def test_list_files_recursive(tmp_path, extensions, nfiles, is_recursive):
    """Test list_files with different input types."""
    # create a temporary txt and json file
    os.makedirs(tmp_path / "folder1", exist_ok=True)
    file_txt = tmp_path / "folder1" / "test.txt"
    file_json = tmp_path / "test.json"
    file_txt.write_text("a\nb\nc\n", encoding="utf-8")
    file_json.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")

    # assertions
    test = list_files(tmp_path, extensions, recursive=is_recursive)
    assert len(test) == nfiles
