"""Unit tests for functions in categorical_plots.py."""

from dscience_toolbox.visualisation import barplot


def test_barplot(titanic_df, tmp_path):
    """Tests the function barplot."""
    # check file is create
    file_path = tmp_path / "barplot.png"
    barplot(titanic_df, "sex", "age", save_path=file_path)
    assert file_path.exists()

    # check Axes object is returned
    ax = barplot(titanic_df, "sex", "age")
    assert ax is not None
