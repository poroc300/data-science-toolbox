"""Functions to handle unstructured data."""

import json
from pathlib import Path


def load_json(path):
    """
    Read a JSON file into a Python dictionary.

    Parameters
    ----------
    path : str or pathlib.PurePath
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_json(data, path, indent=4, ensure_ascii=False):
    """
    Save a dict (or JSON-serializable object) to a JSON file.

    Parameters
    ----------
    data : dict
        Data to serialize into JSON.
    path : str or pathlib.Path
        Output file path.
    indent : int, optional
        Number of spaces for pretty-printing. Default is 4.

    """
    path = Path(path)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=indent)
