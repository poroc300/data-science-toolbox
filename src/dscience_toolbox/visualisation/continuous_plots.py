"""Functions to create continuous plots."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.polynomial import Polynomial

from dscience_toolbox.utils import (
    _check_plotting_inputs,
    _merge_plot_kwargs,
    _validate_pandas_dataframe,
)

# seaborn plot styling
sns.set_theme()


def scatterplot(
    df,
    xvar,
    yvar,
    save_path=None,
    plot_size=(12, 8),
    grp_var=None,
    title=None,
    xlims=None,
    ylims=None,
    font_sizes=None,
    hline=None,
    vline=None,
    rotate_x_labels=0,
    axes_names=None,
    point_size=60,
    regression=None,
    regression_by_group=False,
    plot_kwargs=None,
    reg_kwargs=None,
):
    """
    Create a scatterplot. Uses sns.scatterplot internally, but contains a set of arguments
    that facilitate changing key aesthetic elements and optionally adding regression lines.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    xvar : str
        Name of the numerical column in df to be plotted in x axis.
    yvar : str
        Name of the numerical column in df to be plotted in the y axis.
    save_path : str or pathlib.PurePath, optional
        Path to save the plot. The default is None, which won't save any image
        and will return the matplotlib.axes.Axes object for further editing
        if required.
    plot_size : sequence-like, optional
        Sequence with the width and height of the plot. The default is (12, 8).
    grp_var : str, optional
        Name of the column in df to group values (equivalent to the "hue" parameter
        in seaborn). If supplied, will trigger the creation of a legend. The default
        is None (don't use any variable).
    title : str, optional
        Title of plot. The default is None (no title).
    xlims : sequence-like, optional
        Sequence with the minimum and maximum values for the x axis. The default
        is None (determined by seaborn).
    ylims : sequence-like, optional
        Sequence with the minimum and maximum values for the y axis. The default
        is None (determined by seaborn).
    font_sizes : dict, optional
         Font sizes (int) for plot elements:
             - "title" -> default is 26
             - "axis" -> default is 22
             - "ticks" -> default is 18
             - "legend" -> default is 18
    hline : sequence-like, optional
        Adds a horizontal line. The default is None (no line). Must be a sequence
        with four elements:
            1. y coordinate to draw the line (int)
            2. line style ('-', '--', '-.', ':') -> see matplotlib docs
            3. line width (int)
            4. line colour (str). The default is None
    vline : sequence-like, optional
        Adds a vertical line. The default is None (no line). Must be a sequence
        with four elements:
            1. x coordinate to draw the line (int)
            2. line style ('-', '--', '-.', ':') -> see matplotlib docs
            3. line width (int)
            4. line colour (str). The default is None
    rotate_x_labels : int, optional
        Degree of rotation of x tick labels. The default is 0 (horizontal).
    axes_names : sequence-like, optional
        Must be a sequence of two elements to set the names of the x and y
        axes. This is a convenience argument. The default is None (use directly
        the names supplied in xvar and yvar).
    point_size : int, optional
        Size of scatterplot points. The default is 60.
    regression : {None, 1, 2}, optional
        Adds a regression line to the plot:
            * 1 -> linear regression (degree 1)
            * 2 -> quadratic regression (degree 2)
            * None -> no regression line (default)
    regression_by_group : bool, optional
        If True and grp_var is not None, fits a separate regression line for each group.
        If False, fits a single global regression line. The default is False.
    plot_kwargs : dict, optional
       Additional keyword arguments passed directly to the underlying seaborn
       sns.scatterplot function. The default is None (don't pass any).
    reg_kwargs : dict, optional
        Additional keyword arguments passed to the regression line (matplotlib
        ax.plot, i.e. Line2D styling). The default is None (uses default styling).

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

    # create figure
    plt.figure(figsize=plot_size)

    # create dict with parameters to be added to seaborn function
    params = {
        "x": xvar,
        "y": yvar,
        "data": df,
        "s": point_size,
    }
    if grp_var is not None:
        params["hue"] = grp_var

    # merge params and plot_kwargs
    params = _merge_plot_kwargs(params, plot_kwargs)

    # create visualisation
    ax = sns.scatterplot(**params)

    # fit regression lines
    if regression is not None:
        # invalid input
        if regression not in (1, 2):
            raise ValueError("degree of regression must be 1 or 2")

        # keyword arguments for regression line
        if reg_kwargs is None:
            reg_kwargs = {}
        elif not isinstance(reg_kwargs, dict):
            raise TypeError("reg_kwargs must be a dict")

        # default regression styling and combine with supplied styling
        default_reg_kwargs = {
            "linewidth": 2,
            "linestyle": "-",
            "alpha": 0.8,
            "color": "red",
        }
        reg_kwargs = {**default_reg_kwargs, **reg_kwargs}

        # regression line per group
        if regression_by_group and grp_var is not None:
            for grp in df[grp_var].dropna().unique():
                # get arrays of a group
                subset_df = df.loc[df[grp_var] == grp, [xvar, yvar]].copy()
                x_arr = subset_df[xvar].to_numpy()
                y_arr = subset_df[yvar].to_numpy()

                # fit regression and get predicted values to be plotted as a line
                poly = Polynomial.fit(x_arr, y_arr, deg=regression)
                x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
                y_line = poly(x_line)
                ax.plot(x_line, y_line, **reg_kwargs)

        # global regression line
        else:
            x_arr = df[xvar].to_numpy()
            y_arr = df[yvar].to_numpy()
            poly = Polynomial.fit(x_arr, y_arr, deg=regression)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
            y_line = poly(x_line)
            ax.plot(x_line, y_line, **reg_kwargs)

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

    # set axis limits
    if ylims is not None:
        if len(ylims) != 2:
            raise ValueError("ylims must have 2 elements")
        ax.set_ylim(*ylims)

    if xlims is not None:
        if len(xlims) != 2:
            raise ValueError("xlims must have 2 elements")
        ax.set_xlim(*xlims)

    # add horizontal line
    if hline is not None:
        if len(hline) != 4:
            raise ValueError("hlines must be (y, linestyle, linewidth, color)")
        y, ls, lw, color = hline
        ax.axhline(y=y, linestyle=ls, linewidth=lw, color=color)

    # add vertical line
    if vline is not None:
        if len(vline) != 4:
            raise ValueError("vlines must be (x, linestyle, linewidth, color)")
        x, ls, lw, color = vline
        ax.axvline(y=x, linestyle=ls, linewidth=lw, color=color)

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


def histplot(
    df,
    xvar,
    save_path=None,
    plot_size=(12, 8),
    grp_var=None,
    title=None,
    xlims=None,
    ylims=None,
    font_sizes=None,
    hline=None,
    vline=None,
    rotate_x_labels=0,
    axes_names=None,
    bins="auto",
    binwidth=None,
    stat="count",
    kde=False,
    cumulative=False,
    fill=True,
    element="bars",
    plot_kwargs=None,
):
    """
    Create a histogram. Uses sns.histplot internally, but contains a set of arguments
    that facilitate changing key aesthetic elements and optionally overlaying a KDE curve.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    xvar : str
        Name of the column in df to be plotted on the x axis.
    save_path : str or pathlib.PurePath, optional
        Path to save the plot. The default is None, which won't save any image
        and will return the matplotlib.axes.Axes object for further editing
        if required.
    plot_size : sequence-like, optional
        Sequence with the width and height of the plot. The default is (12, 8).
    grp_var : str, optional
        Name of the column in df to group values (equivalent to the "hue" parameter
        in seaborn). If supplied, will trigger the creation of a legend. The default
        is None (don't use any variable).
    title : str, optional
        Title of plot. The default is None (no title).
    xlims : sequence-like, optional
        Sequence with the minimum and maximum values for the x axis. The default
        is None (determined by seaborn).
    ylims : sequence-like, optional
        Sequence with the minimum and maximum values for the y axis. The default
        is None (determined by seaborn).
    font_sizes : dict, optional
        Font sizes (int) for plot elements:
            - "title"  -> default is 26
            - "axis"   -> default is 22
            - "ticks"  -> default is 18
            - "legend" -> default is 18
    hline : sequence-like, optional
        Adds a horizontal line. The default is None (no line). Must be a sequence
        with four elements:
            1. y coordinate to draw the line (int)
            2. line style ('-', '--', '-.', ':') -> see matplotlib docs
            3. line width (int)
            4. line colour (str)
    vline : sequence-like, optional
        Adds a vertical line. The default is None (no line). Must be a sequence
        with four elements:
            1. x coordinate to draw the line (int)
            2. line style ('-', '--', '-.', ':') -> see matplotlib docs
            3. line width (int)
            4. line colour (str)
    rotate_x_labels : int, optional
        Degree of rotation of x tick labels. The default is 0 (horizontal).
    axes_names : sequence-like, optional
        Must be a sequence of two elements to set the names of the x and y
        axes. This is a convenience argument. The default is None (uses xvar
        for the x axis and the value of stat for the y axis).
    bins : int, str, or sequence-like, optional
        Specification of histogram bins. Accepts an integer number of bins, a
        string rule recognised by numpy (e.g. "auto", "fd", "sturges"), or an
        explicit sequence of bin edges. Mutually exclusive with binwidth —
        only one of the two should be supplied at a time. The default is
        "auto".
    binwidth : float or sequence-like, optional
        Width of each bin in data units, as an alternative to specifying the
        number of bins via bins. Seaborn uses this value to compute the bin
        edges directly, so the resulting number of bins depends on the range
        of the data. A smaller value produces more bins with finer resolution;
        a larger value produces fewer, coarser bins. Accepts a single float
        applied to all groups, or a sequence of floats when grp_var is
        supplied and each group should use a different width. Mutually
        exclusive with bins — if both are supplied, binwidth takes precedence.
        The default is None (use bins instead).
    stat : str, optional
        Aggregate statistic to compute in each bin. Options are:
            - "count"    -> number of observations (default)
            - "frequency"-> count divided by bin width
            - "density"  -> normalised so the total area equals 1
            - "probability" / "proportion" -> normalised so the total height equals 1
            - "percent"  -> like proportion but scaled to 100
    kde : bool, optional
        If True, overlays a Kernel Density Estimate curve on the histogram.
        The default is False.
    cumulative : bool, optional
        If True, plots the cumulative distribution. The default is False.
    fill : bool, optional
        If True, fills the bars/steps with colour. The default is True.
    element : str, optional
        Visual representation of the bins. Controls the geometric mark used to
        display each bin. Options are:
            - "bars" -> standard filled rectangles anchored to the x axis;
                        the most familiar histogram form (default)
            - "step" -> an unfilled staircase outline drawn around the bins;
                        lighter visually and useful when overlaying multiple
                        distributions because it avoids obscuring underlying data
            - "poly" -> a filled polygon connecting the midpoints of each bin
                        top; intermediate between a histogram and a KDE curve,
                        giving a smoother appearance while still binning the data
    plot_kwargs : dict, optional
        Additional keyword arguments passed directly to the underlying seaborn
        sns.histplot function. The default is None (don't pass any).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object that can be further used to modify the plot aesthetics. Will
        only return this object if save_path=None, otherwise an image is created
        and nothing will be returned.
    """
    # validation of input
    df = _validate_pandas_dataframe(df, input_arg="df")
    required_cols = {xvar}
    if grp_var is not None:
        required_cols.add(grp_var)
    _check_plotting_inputs(df, required_cols)

    # create figure
    plt.figure(figsize=plot_size)

    # create dict with parameters to be added to seaborn function
    params = {
        "x": xvar,
        "data": df,
        "bins": bins,
        "stat": stat,
        "kde": kde,
        "cumulative": cumulative,
        "fill": fill,
        "element": element,
    }
    if binwidth is not None:
        params["binwidth"] = binwidth
    if grp_var is not None:
        params["hue"] = grp_var

    # merge params and plot_kwargs
    params = _merge_plot_kwargs(params, plot_kwargs)

    # create histogram
    ax = sns.histplot(**params)

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
        x_label, y_label = xvar, stat.title()
    else:
        if len(axes_names) != 2:
            raise ValueError("axes_names must have 2 elements")
        x_label, y_label = axes_names
    ax.set_xlabel(x_label, fontsize=font_sizes["axis"], labelpad=10)
    ax.set_ylabel(y_label, fontsize=font_sizes["axis"], labelpad=10)
    ax.tick_params(axis="x", labelsize=font_sizes["ticks"], rotation=rotate_x_labels)
    ax.tick_params(axis="y", labelsize=font_sizes["ticks"])

    # set axis limits
    if ylims is not None:
        if len(ylims) != 2:
            raise ValueError("ylims must have 2 elements")
        ax.set_ylim(*ylims)

    if xlims is not None:
        if len(xlims) != 2:
            raise ValueError("xlims must have 2 elements")
        ax.set_xlim(*xlims)

    # add horizontal line
    if hline is not None:
        if len(hline) != 4:
            raise ValueError("hlines must be (y, linestyle, linewidth, color)")
        y, ls, lw, color = hline
        ax.axhline(y=y, linestyle=ls, linewidth=lw, color=color)

    # add vertical line
    if vline is not None:
        if len(vline) != 4:
            raise ValueError("vlines must be (x, linestyle, linewidth, color)")
        x, ls, lw, color = vline
        ax.axvline(x=x, linestyle=ls, linewidth=lw, color=color)

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
