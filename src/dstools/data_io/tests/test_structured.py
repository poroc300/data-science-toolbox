"""Unit tests for functions in structured.py."""

import pandas as pd

from dstools.data_io import load_txt


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
