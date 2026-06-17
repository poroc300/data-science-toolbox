"""Unit tests for functions in categorical_plots.py."""

from dstools.visualisation import barplot, boxplot


def test_barplot(titanic_df, tmp_path):
    """Test the function barplot."""
    # check file is create
    file_path = tmp_path / "barplot.png"
    barplot(titanic_df, "sex", "age", save_path=file_path)
    assert file_path.exists()

    # check Axes object is returned
    ax = barplot(titanic_df, "sex", "age")
    assert ax is not None


def test_boxplot(titanic_df, tmp_path):
    """Test the function boxplot."""
    # check file is create
    file_path = tmp_path / "boxplot.png"
    boxplot(titanic_df, "sex", "age", save_path=file_path)
    assert file_path.exists()

    # check Axes object is returned
    ax = boxplot(titanic_df, "sex", "age")
    assert ax is not None
