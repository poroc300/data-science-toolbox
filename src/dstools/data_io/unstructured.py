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
