"""Functions to support data input/output utilities."""

import json
from pathlib import Path

import pandas as pd


def load_txt(path, as_frame=False, pd_kwargs=None):
    """
    Read a text file either as raw lines or as a pandas DataFrame.

    Parameters
    ----------
    path : str or pathlib.PurePath
        Path to the txt file.
    as_frame : bool, optional
        If True, reads file using pandas.read_csv. If False, returns list of lines.
    pd_kwargs : dict, optional
        Additional arguments passed to pandas.read_csv if as_frame=True. Defaults
        to {"sep": "\t", "header": None}.

    Returns
    -------
    list or DataFrame
        Content of the txt file.
    """
    # load as DataFrame
    path = Path(path)
    if as_frame:
        if pd_kwargs is None:
            pd_kwargs = {"sep": "\t", "header": None}
        return pd.read_csv(path, **pd_kwargs)
    # load as a list with raw lines
    with open(path, "r", encoding="utf-8") as file:
        return file.read().splitlines()


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


def list_files(path, extensions, recursive=False):
    """
    List paths of files that match supplied file extensions (e.g. .json, .csv, etc...).

    Parameters
    ----------
    path : pathlib.PurePath
        Directory to look for files with supplied extensions.
    extensions : str or sequence-like
        Sequence containing file extensions. Extensions can be supplied with leading
        dot (e.g. ".csv") or without it (e.g. "csv"). A string can be supplied as
        well which is internally converted to a list.
    recursive : bool, optional
        Whether to perform a recursive search in path or not. The default is False
        (non-recursive search).

    Returns
    -------
    list
        list with files with the matched extensions.
    """
    # input coercion
    path = Path(path)
    if isinstance(extensions, str):
        extensions = [extensions]

    # list files
    search = Path(path).rglob if recursive else Path(path).glob
    files = []
    for ext in extensions:
        for file in search(f"*.{ext.lstrip('.')}"):
            files.append(file)
    return files
