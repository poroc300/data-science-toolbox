"""Unit tests for functions in data_io.py."""

import json
import os

import pandas as pd
import pytest

from dscience_toolbox.utils import list_files, load_json, load_txt


def test_load_txt_raw_lines(tmp_path):
    """Test loading raw lines as list."""
    # create temporary test file
    file = tmp_path / "test.txt"
    file.write_text("a\nb\nc\n", encoding="utf-8")
    test_lines = load_txt(file)
    assert isinstance(test_lines, list)
    assert test_lines == ["a", "b", "c"]


def test_load_txt_dataframe(tmp_path):
    """Test loading txt as a pandas DataFrame."""
    file = tmp_path / "test.txt"
    file.write_text("a,b\nc,d\n", encoding="utf-8")
    df = load_txt(file, as_frame=True, pd_kwargs={"sep": ",", "header": None})
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_json(tmp_path):
    """Test function load_json."""
    file = tmp_path / "test.json"
    file.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    data = load_json(file)
    assert isinstance(data, dict)
    assert data == {"a": 1, "b": 2}


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
