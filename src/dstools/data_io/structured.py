"""Functions to handle structured data."""

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
