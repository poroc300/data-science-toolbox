"""Helper functions to perform miscellaneous operations on files and paths."""

from pathlib import Path


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
    search = path.rglob if recursive else path.glob
    files = []
    for ext in extensions:
        for file in search(f"*.{ext.lstrip('.')}"):
            files.append(file)
    return files
