"""Unit tests for functions in data_io.py."""

import json

import pandas as pd

from dscience_toolbox.utils import load_json, load_txt


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
