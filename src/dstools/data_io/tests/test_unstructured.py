"""Unit tests for functions in unstructured.py."""

import json

from dstools.data_io import load_json


def test_load_json(tmp_path):
    """Test function load_json."""
    file = tmp_path / "test.json"
    file.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    data = load_json(file)
    assert isinstance(data, dict)
    assert data == {"a": 1, "b": 2}
