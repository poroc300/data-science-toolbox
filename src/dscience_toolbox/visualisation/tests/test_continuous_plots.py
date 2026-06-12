"""Unit tests for functions in categorical_plots.py."""

from dscience_toolbox.visualisation import histplot, scatterplot


def test_scatterplot(sample_df, tmp_path):
    """Test the function scatterplot."""
    # check file is create
    file_path = tmp_path / "scatter.png"
    scatterplot(sample_df, "x1", "x2", save_path=file_path, grp_var="x4")
    assert file_path.exists()

    # check Axes object is returned
    ax = scatterplot(sample_df, "x1", "x2")
    assert ax is not None


def test_histplot(sample_df, tmp_path):
    """Test the function histplot."""
    # check file is create
    file_path = tmp_path / "hist.png"
    histplot(sample_df, "x3", save_path=file_path, grp_var="x4")
    assert file_path.exists()

    # check Axes object is returned
    ax = histplot(sample_df, "x1")
    assert ax is not None
