"""Functions to create categorical plots."""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import is_color_like

from dscience_toolbox.utils import (
    _check_plotting_inputs,
    _merge_plot_kwargs,
    _validate_pandas_dataframe,
)

# seaborn plot styling
sns.set_theme()


def barplot(
    df,
    xvar,
    yvar,
    save_path=None,
    grp_var=None,
    colours=None,
    plot_size=(12, 8),
    title=None,
    ylims=None,
    font_sizes=None,
    rotate_x_labels=0,
    hline=None,
    annots=None,
    axes_names=None,
    plot_kwargs=None,
):
    """
    Create a barplot. Uses sns.barplot internally, but contains a set of arguments
    that facilitate changing key aesthetic elements.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    xvar : str
        Name of column in df to be plotted in x axis.
    yvar : str
        Name of column in df to be plotted in the y axis.
    save_path : str or pathlib.PurePath, optional
        Path to save the plot. The default is None, which won't save any image
        and will return the matplotlib.axes.Axes object for further editing
        if required.
    grp_var : str, optional
        Name of the column in df to group values (equivalent to the "hue" parameter
        in seaborn). If supplied, will trigger the creation of a legend. The default
        is None (don't use any variable).
    colours : str, tuple, or sequence-like, optional
        Colour(s) for the bars. Behaviour depends on the value provided and if
        grp_var is supplied:
            - A single colour recognisable by matplotlib (e.g. "blue", "dodgerblue"
              "#ff5733", (0.2, 0.4, 0.6)) is passed to sns.barplot as argument `color`,
              applying it uniformly to all bars.
            - Any other string (e.g. "Set2", "viridis") that can be interpreted
              by color_palette() or sequence (e.g. a list of colours or a seaborn
              palette object) is passed as `palette`.
        The default is None (seaborn default colours).
    plot_size : sequence-like, optional
        Sequence with the width and height of the plot. The default is (12, 8).
    title : str, optional
        Title of plot. The default is None (no title).
    ylims : sequence-like, optional
        Sequence with the minimum and maximum values for the y axis. The default
        is None (determined by seaborn).
    font_sizes : dict, optional
         Font sizes (int) for plot elements:
             - "title" -> default is 26
             - "axis" -> default is 22
             - "ticks" -> default is 18
             - "legend" -> default is 18
             - "annotations" -> default is 16
    rotate_x_labels : int, optional
        Degree of rotation of x tick labels. The default is 0 (horizontal).
    hline : sequence-like, optional
        Adds a horizontal line. The default is None (no line). Must be a sequence
        with four elements:
            1. y coordinate to draw the line (int)
            2. line style ('-', '--', '-.', ':') -> see matplotlib docs
            3. line width (int)
            4. line colour (str). The default is None
    annots : sequence-like, optional
        Sequence with the labels to be added to the top of each bar. Annotations
        follow the order of rendered bar. If grp_var is supplied, the labels will
        be annotated based on group order. For example, assume grp_var has two
        categories (4 bars). The user must supply first all labels for the first
        category, then for second, etc... The default is None (no annotation of labels).
    axes_names : sequence-like, optional
        Must be a sequence of two elements to set the names of the x and y
        axes. This is a convenience argument. The default is None (use directly
        the names supplied in xvar and yvar).
    plot_kwargs : dict, optional
       Additional keyword arguments passed directly to the underlying seaborn
       sns.barplot function. The default is None (don't pass any).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object that can be further used to modify the plot aesthetics. Will
        only return this object if save_path=None, otherwise an image is created
        and nothing will be returned.

    """
    # validation of input
    df = _validate_pandas_dataframe(df, input_arg="df")
    required_cols = {xvar, yvar}
    if grp_var is not None:
        required_cols.add(grp_var)
    _check_plotting_inputs(df, required_cols)

    # create dict with parameters to be added to the seaborn function
    plt.figure(figsize=plot_size)
    params = {"x": xvar, "y": yvar, "data": df}
    if grp_var is not None:
        params["hue"] = grp_var

    # set parameter for colours
    if colours is not None:
        if isinstance(colours, str) and is_color_like(colours):
            params["color"] = colours
        else:
            params["palette"] = colours

    # merge params and plot_kwargs
    params = _merge_plot_kwargs(params, plot_kwargs)

    # create visualisation
    ax = sns.barplot(**params)

    # create dict with font sizes (will update defaults where only necessary)
    default_fonts = {
        "title": 26,
        "axis": 22,
        "ticks": 18,
        "legend": 18,
        "annotations": 16,
    }
    if font_sizes is None:
        font_sizes = {}
    elif not isinstance(font_sizes, dict):
        raise TypeError("font_sizes must be a dict")
    font_sizes = {**default_fonts, **font_sizes}

    # plot's title
    if title is not None:
        ax.set_title(title, fontsize=font_sizes["title"])

    # axes and ticks labels
    if axes_names is None:
        x_label, y_label = xvar, yvar
    else:
        if len(axes_names) != 2:
            raise ValueError("axes_names must have 2 elements")
        x_label, y_label = axes_names
    ax.set_xlabel(x_label, fontsize=font_sizes["axis"], labelpad=10)
    ax.set_ylabel(y_label, fontsize=font_sizes["axis"], labelpad=10)
    ax.tick_params(axis="x", labelsize=font_sizes["ticks"], rotation=rotate_x_labels)
    ax.tick_params(axis="y", labelsize=font_sizes["ticks"])

    # set y limits
    if ylims is not None:
        if len(ylims) != 2:
            raise ValueError("ylims must have 2 elements")
        ax.set_ylim(*ylims)

    # add a basic horizontal line
    if hline is not None:
        if len(hline) != 4:
            raise ValueError("hline must be (y, linestyle, linewidth, color)")
        y, ls, lw, color = hline
        ax.axhline(y=y, linestyle=ls, linewidth=lw, color=color)

    # add annotations
    if annots is not None:
        # get no. of bars per x axis group and total no. bars in plot
        containers = ax.containers
        n_bars_group = len(containers[0]) if containers else 0
        expected = len(containers) * n_bars_group
        if len(annots) != expected:
            raise ValueError(f"Expected {expected} annotations, got {len(annots)}")

        # assign labels for each x axis group
        idx = 0
        for container in containers:
            labels = annots[idx : idx + n_bars_group]  # noqa: E203
            ax.bar_label(
                container,
                labels=labels,
                padding=3,
                fontsize=font_sizes["annotations"],
            )
            idx += n_bars_group

    # create legend
    if grp_var is not None:
        # remove default seaborn legend
        if ax.legend_ is not None:
            ax.legend_.remove()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            fontsize=font_sizes["legend"],
            bbox_to_anchor=(1.01, 0.7),
            loc="upper left",
            frameon=True,
        )

    # save plot or just return the ax object
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        return ax


def boxplot(
    df,
    xvar,
    yvar,
    save_path=None,
    grp_var=None,
    colours=None,
    plot_size=(12, 8),
    title=None,
    ylims=None,
    font_sizes=None,
    rotate_x_labels=0,
    hline=None,
    axes_names=None,
    plot_kwargs=None,
):
    """
    Create a boxplot. Uses sns.boxplot internally, but contains a set of arguments
    that facilitate changing key aesthetic elements.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    xvar : str
        Name of column in df to be plotted in x axis.
    yvar : str
        Name of column in df to be plotted in the y axis.
    save_path : str or pathlib.PurePath, optional
        Path to save the plot. The default is None, which won't save any image
        and will return the matplotlib.axes.Axes object for further editing
        if required.
    grp_var : str, optional
        Name of the column in df to group values (equivalent to the "hue" parameter
        in seaborn). If supplied, will trigger the creation of a legend. The default
        is None (don't use any variable).
    colours : str, tuple, or sequence-like, optional
        Colour(s) for the boxplots. Behaviour depends on the value provided and if
        grp_var is supplied:
            - A single colour recognisable by matplotlib (e.g. "blue", "dodgerblue"
              "#ff5733", (0.2, 0.4, 0.6)) is passed to sns.boxplot as argument `color`,
              applying it uniformly to all boxplots.
            - Any other string (e.g. "Set2", "viridis") that can be interpreted
              by color_palette() or sequence (e.g. a list of colours or a seaborn
              palette object) is passed as `palette`.
        The default is None (seaborn default colours).
    plot_size : sequence-like, optional
        Sequence with the width and height of the plot. The default is (12, 8).
    title : str, optional
        Title of plot. The default is None (no title).
    ylims : sequence-like, optional
        Sequence with the minimum and maximum values for the y axis. The default
        is None (determined by seaborn).
    font_sizes : dict, optional
        Font sizes (int) for plot elements:
            - "title" -> default is 26
            - "axis" -> default is 22
            - "ticks" -> default is 18
            - "legend" -> default is 18
    rotate_x_labels : int, optional
        Degree of rotation of x tick labels. The default is 0 (horizontal).
    hline : sequence-like, optional
        Adds a horizontal line. The default is None (no line). Must be a sequence
        with four elements:
            1. y coordinate to draw the line (int)
            2. line style ('-', '--', '-.', ':') -> see matplotlib docs
            3. line width (int)
            4. line colour (str)
    axes_names : sequence-like, optional
        Must be a sequence of two elements to set the names of the x and y
        axes. This is a convenience argument. The default is None (use directly
        the names supplied in xvar and yvar).
    plot_kwargs : dict, optional
        Additional keyword arguments passed directly to the underlying seaborn
        sns.boxplot function. The default is None (don't pass any).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object that can be further used to modify the plot aesthetics. Will
        only return this object if save_path=None, otherwise an image is created
        and nothing will be returned.
    """
    # validation of input
    df = _validate_pandas_dataframe(df, input_arg="df")
    required_cols = {xvar, yvar}
    if grp_var is not None:
        required_cols.add(grp_var)
    _check_plotting_inputs(df, required_cols)

    # create dict with parameters to be added to the seaborn function
    plt.figure(figsize=plot_size)
    params = {"x": xvar, "y": yvar, "data": df}
    if grp_var is not None:
        params["hue"] = grp_var

    # set parameter for colours
    if colours is not None:
        if isinstance(colours, str) and is_color_like(colours):
            params["color"] = colours
        else:
            params["palette"] = colours

    # merge params and plot_kwargs
    params = _merge_plot_kwargs(params, plot_kwargs)

    # create visualisation
    ax = sns.boxplot(**params)

    # create dict with font sizes (will update defaults where only necessary)
    default_fonts = {
        "title": 26,
        "axis": 22,
        "ticks": 18,
        "legend": 18,
    }
    if font_sizes is None:
        font_sizes = {}
    elif not isinstance(font_sizes, dict):
        raise TypeError("font_sizes must be a dict")
    font_sizes = {**default_fonts, **font_sizes}

    # plot's title
    if title is not None:
        ax.set_title(title, fontsize=font_sizes["title"])

    # axes and ticks labels
    if axes_names is None:
        x_label, y_label = xvar, yvar
    else:
        if len(axes_names) != 2:
            raise ValueError("axes_names must have 2 elements")
        x_label, y_label = axes_names
    ax.set_xlabel(x_label, fontsize=font_sizes["axis"], labelpad=10)
    ax.set_ylabel(y_label, fontsize=font_sizes["axis"], labelpad=10)
    ax.tick_params(axis="x", labelsize=font_sizes["ticks"], rotation=rotate_x_labels)
    ax.tick_params(axis="y", labelsize=font_sizes["ticks"])

    # set y limits
    if ylims is not None:
        if len(ylims) != 2:
            raise ValueError("ylims must have 2 elements")
        ax.set_ylim(*ylims)

    # add a basic horizontal line
    if hline is not None:
        if len(hline) != 4:
            raise ValueError("hlines must be (y, linestyle, linewidth, color)")
        y, ls, lw, color = hline
        ax.axhline(y=y, linestyle=ls, linewidth=lw, color=color)

    # create legend
    if grp_var is not None:
        # remove default seaborn legend
        if ax.legend_ is not None:
            ax.legend_.remove()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            fontsize=font_sizes["legend"],
            bbox_to_anchor=(1.01, 0.7),
            loc="upper left",
            frameon=True,
        )

    # save plot or just return the ax object
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        return ax
