"""
Purpose: compare parameters from different programs
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Helper functions
import utilities.plotting.helper_functions as help

# Common python packages
import pandas as pd  # for reading csv
import matplotlib.pyplot as plt  # for plotting
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import matplotlib.patches as mpatches  # for legend
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec  # for subplots
import numpy as np  # for sorting arrays
import math  # for rounding to significant figures


"""
Compare one variable from each program
"""


def plot_overlapping_histograms(
    data_dict,
    programs,
    program_colors,
    selected_variables=None,
    include_cf_variables=False,
    bin_count=200,
    log_scale=False,
    x_label_global="2022 USD",
    x_label_variable=False,
    single_legend_location="center",
    legend_location=(0.9, 0.43),
    symlog_linthresh=1e0,
    remove_zeros=False,
    constrained_layout=False,
    force_bin_count=False,
    title=None,
    subtitle=None,
):
    """
    Plot overlapping histograms for one variable from each program.

    Args:
        data_dict (dict): Dictionary mapping program names to dataframes.
        programs (list): List of program names.
        program_colors (dict): Dictionary mapping program names to colors.
        selected_variables (list): List of variables to plot.
        include_cf_variables (bool): Whether to include the counterfactual variable.
        bin_count (int): Number of bins.
        log_scale (bool): Whether to use a log scale.
        x_label_global (str): Label of the x-axis for the entire plot.
        x_label_variable (bool): Whether to label the x-axis using variable names.
        single_legend_location (str): Location of the legend for the entire plot.
        legend_location (tuple): Location of the legend for each subplot.
        symlog_linthresh (float): Threshold for the symlog scale.
        remove_zeros (bool): Whether to remove zeros from the data.
        constrained_layout (bool): Whether to use constrained layout.
        force_bin_count (bool): Whether to force the number of bins.
        title (str): Title of the plot.
        subtitle (str): Subtitle of the plot.

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot.
    """
    main_variables = selected_variables
    cf_variables = [var for var in selected_variables if include_cf_variables == True]

    ncols = 1 if len(main_variables) == 1 else (3 if len(main_variables) > 4 else 2)
    nrows = int(np.ceil(len(main_variables) / ncols))
    if constrained_layout == False:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 4 * nrows),
            constrained_layout=False,
            subplot_kw=dict(box_aspect=1),
        )
        plt.subplots_adjust(wspace=0)
    else:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 4 * nrows),
        )

    if len(main_variables) == 1:
        axes = [axes]
    elif nrows * ncols > 1:
        axes = axes.reshape(-1)

    legend_patches = [
        Patch(color=color, label=help.prettify_label(label, capitalize_each_word=True))
        for label, color in program_colors.items()
    ]

    for idx, variable in enumerate(main_variables):
        ax = axes[idx]
        min_value = float("inf")
        max_value = float("-inf")
        all_data_between_0_and_1 = True

        for program in programs:
            if variable not in data_dict[program].columns:
                continue

            data = data_dict[program][variable]
            min_value = min(min_value, data.min())
            max_value = max(max_value, data.max())
            if variable in cf_variables:
                if variable + "_cf" not in data_dict[program].columns:
                    continue

                data_cf = data_dict[program][variable + "_cf"]
                min_value = min(min_value, data_cf.min())
                max_value = max(max_value, data_cf.max())

            if not (0 <= data.min() and data.max() <= 1):
                all_data_between_0_and_1 = False

        symlog_scale = False
        if log_scale and (
            min_value <= 0
            or any(min(data[variable]) <= 0 for data in data_dict.values())
        ):
            symlog_scale = True

        log_scale_subplot = not all_data_between_0_and_1 and (log_scale or symlog_scale)

        # log_scale_subplot = not all_data_between_0_and_1 and log_scale

        if log_scale_subplot:
            bins = np.logspace(
                np.log10(min_value + 1e-9), np.log10(max_value), bin_count
            )
            ax.set_xscale("log")
        else:
            if all_data_between_0_and_1:
                bins = np.linspace(0, np.ceil(max_value * 10) / 10, bin_count)
            else:
                # Ensure that bin_count is no greater than the range of values
                if force_bin_count == False:
                    bin_count = min(bin_count, int(max_value - min_value) + 1)

                bins = np.linspace(min_value, max_value, bin_count)

        for program in programs:
            if variable not in data_dict[program].columns:
                continue

            data = data_dict[program][variable]

            # If remove_zeros is True, filter out zeros from the data
            if remove_zeros:
                data = data[data != 0]

            ax.hist(
                data,
                bins=bins,
                alpha=0.5,
                color=program_colors[program],
                label=help.prettify_label(program, capitalize_each_word=True),
            )

            if variable in cf_variables:
                if variable + "_cf" not in data_dict[program].columns:
                    continue

                data_cf = data_dict[program][variable + "_cf"]

                # If remove_zeros is True, filter out zeros from the counterfactual data
                if remove_zeros:
                    data_cf = data_cf[data_cf != 0]

                ax.hist(
                    data_cf,
                    bins=bins,
                    alpha=0.25,
                    color=program_colors[program],
                    label=program + " (counterfactual)",
                )

        if log_scale_subplot:
            if symlog_scale:
                ax.set_xscale("symlog", linthresh=symlog_linthresh)
                ax.set_xlim(left=0)

            else:
                ax.set_xscale("log")
                xticks = help.generate_xticks(min_value, max_value)
                ax.get_xaxis().set_ticks([])
                ax.xaxis.set_major_locator(FixedLocator(xticks))
                if len(xticks) <= 4:
                    min_value = min(xticks)
                    max_value = max(xticks)
                ax.set_xlim(min_value, max_value)

        if all_data_between_0_and_1:
            xlim_right = np.minimum(1, np.ceil(max_value * 10) / 10)
            ax.set_xlim(left=0, right=xlim_right)

        # Comment out the following line to remove y labels
        # ax.set_ylabel('Frequency')

        # Hide the y-axis and its components
        ax.yaxis.set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if idx // ncols == nrows - 1:
            ax.set_xlabel(x_label_global)

        if x_label_variable == True:
            pretty_title = help.prettify_label(f"{variable}", capitalize_each_word=True)
            split_pretty_title = help.split_title(pretty_title, max_length=30)
            ax.set_xlabel(split_pretty_title)
        else:
            pretty_title = help.prettify_label(f"{variable}", capitalize_each_word=True)
            split_pretty_title = help.split_title(pretty_title, max_length=30)
            ax.set_title(split_pretty_title)

    # Add legend to the side
    if nrows * ncols > 1:
        if len(main_variables) % ncols == 0:
            # fig.subplots_adjust(bottom=0.15)
            fig.legend(
                handles=legend_patches,
                loc=legend_location,
                ncol=1,
                bbox_to_anchor=legend_location,
            )
        else:
            ax = axes[-1]
            ax.axis("off")
            if len(main_variables) <= 4:
                legend_cols = len(programs)
            else:
                legend_cols = len(programs) // 3
            ax.legend(handles=legend_patches, loc="center", ncol=legend_cols)
    else:
        fig.subplots_adjust(bottom=0.2)
        fig.legend(
            handles=legend_patches,
            loc=single_legend_location,
            ncol=1,
            bbox_to_anchor=legend_location,
        )

    if title:  # Add this check to add title if it's provided
        plt.suptitle(title, fontsize=14, y=1.05)
    if subtitle:
        fig.text(
            0.5, 0.98, subtitle, ha="center", va="center", fontsize=12, style="italic"
        )

    plt.tight_layout()

    # Return figure
    return fig


"""
Pipeline probabilities plot
"""


def plot_pipeline_probabilities(
    data_dict,
    programs,
    program_colors,
    param_names,
    variables=None,
    selected_variables=None,
    bin_count=200,
    log_scale=False,
    x_label_global="2022 USD",
    x_label_variable=False,
    symlog_linthresh=1e0,
    remove_zeros=False,
    force_many_bins=False,
    title=None,
    subtitle=None,
):
    """
    Plots a grid of histograms where each row corresponds to a program and each column corresponds to a variable.
    Within each subplot, it plots a histogram of the variable for the corresponding program, as well as a histogram
    of the counterfactual of the variable if it exists.

    Args:
        data_dict (dict): A dictionary where keys are program names and values are pandas DataFrames containing the data for each program.
        programs (list): A list of program names.
        program_colors (dict): A dictionary where keys are program names and values are the colors to use for each program.
        param_names (list): A list of parameter names.
        variables (list, optional): A list of variables to include in the plot. If None, all shared variables are included.
        selected_variables (list, optional): A list of selected variables to include in the plot. If None, all variables are included.
        bin_count (int, optional): The number of bins to use in the histograms. Default is 200.
        log_scale (bool, optional): Whether to use a log scale on the x-axis. Default is False.
        x_label_global (str, optional): The label for the x-axis. Default is '2022 USD'.
        x_label_variable (bool, optional): Whether to use the variable name as the x-axis label. Default is False.
        symlog_linthresh (float, optional): The range within which the plot is linear (when using symlog scale). Default is 1e0.
        remove_zeros (bool, optional): Whether to remove zeros from the data. Default is False.
        title (str, optional): The title for the plot. Default is None.
        subtitle (str, optional): The subtitle for the plot. Default is None.

    Returns:
        fig: The matplotlib figure containing the plot.
    """
    if variables is None:
        variables = help.find_shared_variables(data_dict, programs, selected_variables)

    main_variables = [var for var in param_names if not var.endswith("_cf")]
    cf_variables = [var for var in param_names if var + "_cf" in variables]

    if selected_variables != None:
        main_variables = [var for var in main_variables if var in selected_variables]
        cf_variables = [var for var in cf_variables if var in selected_variables]

    # Correct the order of the variables
    main_variables = pd.Categorical(
        main_variables, categories=selected_variables, ordered=True
    )
    main_variables.sort_values(inplace=True)
    cf_variables = [var + "_cf" for var in main_variables]

    ncols = len(main_variables)
    nrows = len(programs)

    # Create a GridSpec with custom width ratios
    gs = gridspec.GridSpec(
        nrows, ncols + 1, width_ratios=[1] * (ncols - 1) + [1.3] + [0.3]
    )  # make the last column twice as wide

    fig = plt.figure(
        figsize=(4 * (ncols + 1), 4 * nrows)
    )  # adjust the figure size to accommodate the wider last column

    # Create the subplots using the GridSpec
    axes = [[plt.subplot(gs[i, j]) for j in range(ncols)] for i in range(nrows)]

    plt.subplots_adjust(wspace=3)

    if nrows == 1:
        axes = axes.reshape(-1, 1)
    if ncols == 1:
        axes = axes.reshape(1, -1)

    for col, variable in enumerate(main_variables):
        max_value_across_programs = max(
            data_dict[program][variable].max()
            for program in programs
            if variable in data_dict[program].columns
        )

        for row, program in enumerate(programs):
            # ax = axes[row, col]
            ax = axes[row][col]
            if variable not in data_dict[program].columns:
                ax.axis("off")
                continue

            data = data_dict[program][variable]
            min_value = data.min()
            max_value = data.max()
            all_data_between_0_and_1 = 0 <= min_value and max_value <= 1

            symlog_scale = False
            if log_scale and min_value <= 0:
                symlog_scale = True

            log_scale_subplot = not all_data_between_0_and_1 and (
                log_scale or symlog_scale
            )

            if log_scale_subplot:
                bins = np.logspace(
                    np.log10(min_value + 1e-9),
                    np.log10(max_value_across_programs),
                    bin_count,
                )
                ax.set_xscale("log")
            else:
                if all_data_between_0_and_1:
                    bins = np.linspace(
                        0, np.ceil(max_value_across_programs * 20) / 20, bin_count
                    )
                else:
                    if force_many_bins == False:
                        bin_count = min(
                            bin_count, int(max_value_across_programs - min_value) + 1
                        )

                    bins = np.linspace(min_value, max_value_across_programs, bin_count)

            if remove_zeros:
                data = data[data != 0]

            ax.hist(
                data,
                bins=bins,
                alpha=0.5,
                color=program_colors[program],
                label="With program",
            )

            if (
                variable + "_cf" in cf_variables
                and variable + "_cf" in data_dict[program].columns
            ):
                data_cf = data_dict[program][variable + "_cf"]
                if remove_zeros:
                    data_cf = data_cf[data_cf != 0]
                ax.hist(
                    data_cf,
                    bins=bins,
                    alpha=0.25,
                    color=program_colors[program],
                    label="Without program",
                )

            if log_scale_subplot:
                if symlog_scale:
                    ax.set_xscale("symlog", linthresh=symlog_linthresh)
                    ax.set_xlim(left=0)
                else:
                    ax.set_xscale("log")
                    xticks = help.generate_xticks(min_value, max_value_across_programs)
                    ax.get_xaxis().set_ticks([])
                    ax.xaxis.set_major_locator(FixedLocator(xticks))
                    if len(xticks) <= 4:
                        min_value = min(xticks)
                        max_value = max(xticks)
                    ax.set_xlim(min_value, max_value_across_programs)
            else:
                xlim_right = np.minimum(1, np.ceil(max_value_across_programs * 20) / 20)
                ax.set_xlim(left=0, right=xlim_right)

            # ax.yaxis.set_visible(False)
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            if row == nrows - 1:
                ax.set_xlabel(x_label_global)

            if row == 0:
                pretty_title = help.prettify_label(f"{variable}")
                split_pretty_title = help.split_title(pretty_title, max_length=25)
                ax.set_title(split_pretty_title)

            if col == 0:
                pretty_label = help.prettify_label(program, capitalize_each_word=True)
                split_pretty_label = help.split_title(pretty_label, 20)
                ax.set_ylabel(split_pretty_label, fontsize=13, rotation=90, labelpad=10)

            # Add a legend to the right of the right-most column
            if col == ncols - 1:
                # ax.legend()
                ax_legend = fig.add_subplot(
                    gs[row, -1]
                )  # add a new subplot for the legend
                ax_legend.axis("off")  # turn off the axis
                (
                    handles,
                    labels,
                ) = (
                    ax.get_legend_handles_labels()
                )  # get the handles and labels from the last plot in the row
                ax_legend.legend(
                    handles, labels, loc="center"
                )  # add the legend to the new subplot

    if title:  # Add this check to add title if it's provided
        plt.suptitle(title, fontsize=14, y=1.05)
    if subtitle:
        fig.text(
            0.5, 0.98, subtitle, ha="center", va="center", fontsize=12, style="italic"
        )

    plt.tight_layout()

    # Return figure
    return fig


"""
Break down source of QARYs by participant and researcher type
"""


# Number of scientist-equivalents of different types
def plot_n_scientist_equivalent(
    df_params,
    programs=["tdc", "neurips_social", "neurips_workshop"],
    researcher_types=["scientist", "professor", "engineer", "phd"],
    participant_types=["contender", "attendee"],
    participant_colors={"contender": "red", "attendee": "blue"},
    bins=200,
    log_scale=True,
    figsize=(15, 9),
    legend_location=None,
    linthresh=0.1,
    title=None,
    subtitle=None,
):
    """
    Plots the n_scientist_equivalent variables for different researcher types and programs.

    Args:
        df_params: Dictionary with DataFrame values for each program
        bins: int, optional, default=200
            Number of bins for histograms
        log_scale: bool, optional, default=True
            Whether to use log scale for x-axis
        figsize: tuple, optional, default=(15, 9)
            Size of the figure
        legend_location: str or None, optional, default=None
            The location of the legend
        title: str or None, optional, default=None
            The title of the plot
        subtitle: str or None, optional, default=None
            The subtitle of the plot

    Returns:
        fig: matplotlib figure
    """

    fig, axes = plt.subplots(
        nrows=len(programs), ncols=len(researcher_types), figsize=figsize, sharex=True
    )

    legend_patches = []

    for row, program in enumerate(programs):
        for col, researcher_type in enumerate(researcher_types):
            ax = axes[row, col]

            for participant_idx, participant_type in enumerate(participant_types):
                participant_color = participant_colors[participant_type]
                # program_color = program_colors[program][participant_idx]

                label_type = f"{participant_type}_{researcher_type}"
                label_scientist_equiv = f"n_scientist_equivalent_{label_type}"
                label_raw = f"n_{label_type}"

                max_val_raw = (
                    df_params[program][label_raw].max()
                    if label_raw in df_params[program].columns
                    else 0
                )
                max_val_scientist_equiv = (
                    df_params[program][label_scientist_equiv].max()
                    if label_scientist_equiv in df_params[program].columns
                    else 0
                )
                max_val = max(max_val_raw, max_val_scientist_equiv)
                bins_range = np.logspace(np.log10(linthresh), np.log10(max_val), bins)

                if label_scientist_equiv in df_params[program].columns:
                    df_params[program][label_scientist_equiv].plot.hist(
                        ax=ax,
                        bins=bins_range,
                        color=participant_color,
                        alpha=0.7,
                        logx=log_scale,
                    )
                if label_raw in df_params[program].columns:
                    df_params[program][label_raw].plot.hist(
                        ax=ax,
                        bins=bins_range,
                        color=participant_color,
                        alpha=0.3,
                        logx=log_scale,
                    )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(row == len(programs) - 1)
            ax.spines["left"].set_visible(False)

            if row == 0:
                ax.set_title(help.prettify_label(researcher_type), fontsize=14)

            if col == 0:
                prettified_program = help.prettify_label(
                    program, capitalize_each_word=True
                )
                split_prettified_program = help.split_title(prettified_program, 15)
                ax.set_ylabel(split_prettified_program, fontsize=14)
            else:
                ax.set_ylabel("")

            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000])
            ax.set_xticklabels(
                [
                    r"$10^-2$",
                    r"$10^-1$",
                    r"$10^0$",
                    r"$10^1$",
                    r"$10^2$",
                    r"$10^3$",
                    r"$10^4$",
                ],
                fontsize=10 if row == len(programs) - 1 else 0,
            )
            ax.set_yticks([])
            ax.set_yticklabels([])

    # Prepare legend patches
    for participant_type, color in participant_colors.items():
        legend_patches.append(
            Patch(
                facecolor=color,
                alpha=0.3,
                label=f"{help.prettify_label(participant_type)}: Raw number",
            )
        )
        legend_patches.append(
            Patch(
                facecolor=color,
                alpha=1.0,
                label=f"{help.prettify_label(participant_type)}: Scientist-equivalent",
            )
        )

    fig.legend(
        handles=legend_patches,
        loc=legend_location if legend_location else "lower center",
        bbox_to_anchor=(1.1, 0.43),
        ncol=1,
    )

    if title:  # Add this check to add title if it's provided
        plt.suptitle(title, fontsize=14, y=1.05)
    if subtitle:
        fig.text(
            0.5, 1, subtitle, ha="center", va="center", fontsize=12, style="italic"
        )

    plt.tight_layout()
    # plt.show()
    return fig


# Number of QARYs per participant and researcher type
def plot_qarys_per_participant(
    df_params,
    programs=["tdc", "neurips_social", "neurips_workshop"],
    researcher_types=["scientist", "professor", "engineer", "phd"],
    program_colors={
        "tdc": {"contender": "lightgreen", "attendee": "green"},
        "neurips_social": {"contender": "plum", "attendee": "purple"},
        "neurips_workshop": {"contender": "moccasin", "attendee": "orange"},
    },
    offset_if_double_label=0,
    bar_label_shift=0,
    bar_label_scale=(1 / 30),
    figsize=(10, 12),
    legend_location=None,
    linthresh=1,
    ymin=1,
    log_scale=True,
):
    """
    Plots stacked bar charts for qarys per participant and researcher type.

    Args:
        df_params: Dictionary with DataFrame values for each program
        researcher_types: list, optional, default=['scientist', 'professor', 'engineer', 'phd']
            List of researcher types
        program_colors: dict, optional, default={'tdc': 'green', 'neurips_social': 'purple', 'neurips_workshop': 'orange'}
            Dictionary with colors for each program
        figsize: tuple, optional, default=(10, 12)
            Size of the figure
        legend_location: str or None, optional, default=None
            The location of the legend

    Returns:
        fig: matplotlib figure
    """

    fig, axes = plt.subplots(nrows=len(programs), ncols=1, figsize=figsize, sharex=True)

    legend_patches = []
    max_y = 0

    for row, program in enumerate(programs):
        ax = axes[row]

        bar_width = 0.35

        for col, researcher_type in enumerate(researcher_types):
            bottom_cf = [0, 0]  # Separate bottom values for contender and attendee

            for idx, participant_type in enumerate(["contender", "attendee"]):
                qarys_cf_col = f"qarys_per_{participant_type}_{researcher_type}_cf"
                qarys_col = f"qarys_per_{participant_type}_{researcher_type}"
                if (
                    qarys_col in df_params[program].columns
                    and qarys_cf_col in df_params[program].columns
                ):
                    qarys_cf = df_params[program][qarys_cf_col].mean()
                    qarys = df_params[program][qarys_col].mean()
                    qarys_diff = qarys - qarys_cf
                    max_y = max(max_y, qarys_cf + qarys_diff)

                    # Plot the counterfactual bar segment
                    ax.bar(
                        col + idx * bar_width,
                        qarys_cf,
                        bottom=bottom_cf[idx],
                        color=program_colors[program][participant_type],
                        alpha=0.5,
                        width=bar_width,
                    )
                    bottom_cf[idx] += qarys_cf

                    # Plot the actual minus counterfactual bar segment
                    ax.bar(
                        col + idx * bar_width,
                        qarys_diff,
                        bottom=bottom_cf[idx],
                        color=program_colors[program][participant_type],
                        alpha=0.7 if participant_type == "contender" else 1,
                        width=bar_width,
                    )
                    bottom_cf[idx] += qarys_diff

                # Add the participant type label on top of the bars in the first (row == 0) and second (row == 1) subplots
                if row == 0 and idx == 0:
                    ax.text(
                        col + idx * bar_width,
                        (bottom_cf[idx] * bar_label_scale)
                        + bar_label_shift
                        + qarys_cf
                        + qarys_diff,
                        "Contender",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                elif row == 1 and idx == 1:
                    ax.text(
                        col + idx * bar_width,
                        (bottom_cf[idx] * bar_label_scale)
                        + bar_label_shift
                        + qarys_cf
                        + qarys_diff,
                        "Attendee",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                elif row == 2 and idx == 0:
                    ax.text(
                        col + idx * bar_width - offset_if_double_label,
                        (bottom_cf[idx] * bar_label_scale)
                        + bar_label_shift
                        + qarys_cf
                        + qarys_diff,
                        "Contender",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                elif row == 2 and idx == 1:
                    ax.text(
                        col + idx * bar_width + offset_if_double_label,
                        (bottom_cf[idx] * bar_label_scale)
                        + bar_label_shift
                        + qarys_cf
                        + qarys_diff,
                        "Attendee",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # ax.spines['bottom'].set_visible(row == len(programs) - 1)

        ax.set_ylabel(help.prettify_label(program), fontsize=14)
        ax.set_xticks(np.arange(len(researcher_types)) + bar_width / 2)
        ax.set_xticklabels(
            [help.prettify_label(rt) for rt in researcher_types],
            fontsize=10 if row == len(programs) - 1 else 0,
        )

        # Conditionally apply log scale
        if log_scale:
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_yticks([0.1, 1, 10, 100])
            ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])

            # Round up max_y to the nearest power of 10
            max_y = 10 ** math.ceil(math.log10(max_y))

            for program in programs:
                ax = axes[programs.index(program)]

                # Set ylim for each subplot
                ax.set_ylim(ymin, max_y * 1.1)

    for program in programs:
        ax = axes[programs.index(program)]

        # Set ylim for each subplot
        ax.set_ylim(ymin, max_y * 1.1)

    plt.tight_layout()
    return fig


def plot_counterfactual_n_scientist_equivalent(
    data_dict,
    programs,
    program_colors,
    selected_variables,
    bin_count=200,
    x_label_global="2022 USD",
    title=None,
    subtitle=None,
    pseudolog=False,
):
    """
    Plots number of scientist equivalents for different participant types and
        programs, with or without the program taking place.

    Args:
        data_dict (dict): Dictionary of dataframes, with keys corresponding to
            program names and values corresponding to dataframes containing
            program parameters.
        programs (list): List of program names to plot.
        program_colors (dict): Dictionary of program colors, with keys
            corresponding to program names and values corresponding to colors.
        selected_variables (list): List of variables to plot.
        bin_count (int): Number of bins to use for histogram.
        x_label_global (str): Label for x-axis.
        title (str): Title for plot.
        subtitle (str): Subtitle for plot.
        pseudolog (bool): Whether to use pseudolog scale for y-axis.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """
    main_variables = selected_variables

    ncols = len(main_variables)
    nrows = len(programs)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
    )

    if nrows == 1:
        axes = axes.reshape(-1, 1)
    if ncols == 1:
        axes = axes.reshape(1, -1)

    for col, variable in enumerate(main_variables):
        min_value_across_programs = min(
            data_dict[program][variable].min()
            for program in programs
            if variable in data_dict[program].columns
        )

        try:
            linthresh = 10 ** np.floor(np.log10(min_value_across_programs))
        except:
            linthresh = 1e-2
        linthresh = np.max([linthresh, 1e-4])

        max_value_across_programs = max(
            data_dict[program][variable].max()
            for program in programs
            if variable in data_dict[program].columns
        )

        for row, program in enumerate(programs):
            ax = axes[row][col]

            data = data_dict[program].get(variable)
            data_cf = data_dict[program].get(variable + "_cf")

            # Create bins for histogram
            if pseudolog:
                bins = np.concatenate(
                    (
                        [0.0],
                        np.logspace(
                            np.log10(linthresh),
                            np.log10(max_value_across_programs),
                            bin_count,
                        ),
                    )
                )
            else:
                bins = np.linspace(
                    0, np.ceil(max_value_across_programs * 20) / 20, bin_count
                )

            # Create histograms or empty plot
            if data is not None:
                ax.hist(
                    data,
                    bins=bins,
                    alpha=0.5,
                    color=program_colors[program],
                    label="With program",
                )

            if data_cf is not None:
                ax.hist(
                    data_cf,
                    bins=bins,
                    alpha=0.25,
                    color=program_colors[program],
                    label="Without program",
                )

            if (
                data is None and data_cf is None
            ):  # Added this condition to handle empty plot
                ax.set_xlim(left=0, right=max_value_across_programs)
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_xlabel(x_label_global)

                if row == 0:
                    pretty_title = help.prettify_label(f"{variable}")
                    split_pretty_title = help.split_title(pretty_title, max_length=30)
                    ax.set_title(split_pretty_title)

                continue

            ax.set_xlim(left=0, right=max_value_across_programs)
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            if pseudolog:
                ax.set_xscale("symlog", linthresh=linthresh)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(
                        lambda x, pos: f"$10^{{{int(np.log10(x))}}}$"
                        if x >= 1
                        else f"{x}"
                    )
                )

            if row == nrows - 1:
                ax.set_xlabel(x_label_global)

            if row == 0:
                pretty_title = help.prettify_label(f"{variable}")
                split_pretty_title = help.split_title(pretty_title, max_length=30)
                ax.set_title(split_pretty_title)

            if col == 0:
                pretty_label = help.prettify_label(program, capitalize_each_word=True)
                split_pretty_label = help.split_title(pretty_label, 20)
                ax.set_ylabel(split_pretty_label, fontsize=13, rotation=90, labelpad=10)

    if title:  # Add this check to add title if it's provided
        plt.suptitle(title, fontsize=14, y=1.05)
    if subtitle:
        fig.text(
            0.5, 0.98, subtitle, ha="center", va="center", fontsize=12, style="italic"
        )

    plt.tight_layout()

    # Return figure
    return fig


"""
Compare one variable from each program, alongside its counterfactual
"""


def plot_overlapping_histograms_cf(
    data_dict,
    programs,
    variable,
    bin_count=200,
    program_colors=None,
    same_x_scale=True,
    log_scale=True,
    symlog_threshold=1,
    xlim_bottom=None,
    alpha=0.5,
    alpha_cf=0.2,
    single_legend=False,
    legend_locations=None,
    title=None,
    subtitle=None,
):
    """
    Plot overlapping histograms comparing the variable distribution
    of different programs against their counterfactuals.

    Args:
        data_dict (dict): Dictionary of pandas DataFrames, where keys are program names and values are DataFrames.
        programs (list): List of program names to be plotted.
        variable (str): The name of the variable to be plotted.
        bin_count (int): Number of bins for the histograms. Default is 200.
        program_colors (dict): Dictionary of colors for each program. Default is None, and colors will be selected automatically.
        same_x_scale (bool): If True, the x-axis scale will be the same for all plots. Default is True.
        log_scale (bool): If True, the x-axis will be displayed on a logarithmic scale. Default is True.
        symlog_threshold (float): Threshold for symlog scale. Default is 1.
        xlim_bottom (float): Optional lower limit for the x-axis. Default is None, which means it will be determined automatically.
        alpha (float): Alpha value (transparency) for the main histograms. Default is 0.8.
        alpha_cf (float): Alpha value (transparency) for the counterfactual histograms. Default is 0.5.
        single_legend (bool): If True, display a single legend for all plots. Default is False.
        legend_locations (list): List of legend locations for each program plot. Default is None, and locations will be set to 'best'.

    Returns:
        fig (matplotlib.figure.Figure): The figure containing the plot.
    """
    if program_colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        program_colors = {program: color for program, color in zip(programs, colors)}

    fig, axes = plt.subplots(
        nrows=len(programs), ncols=1, figsize=(6, 4 * len(programs))
    )

    if len(programs) == 1:
        axes = [axes]  # ensure axes is a list for consistency

    variable_cf = variable + "_cf"

    # Calculate global min and max values if same_x_scale is True
    if same_x_scale:
        min_value = float("inf")
        max_value = float("-inf")
        for program in programs:
            min_value = min(
                min_value,
                data_dict[program][variable].min(),
                data_dict[program][variable_cf].min(),
            )
            max_value = max(
                max_value,
                data_dict[program][variable].max(),
                data_dict[program][variable_cf].max(),
            )

    if legend_locations is None:
        legend_locations = ["best"] * len(programs)

    # Determine the appropriate scale for x-axis based on the global minimum value
    x_scale = "linear"
    if log_scale:
        if min_value <= 0:
            x_scale = "symlog"
        else:
            x_scale = "log"

    for idx, (ax, program) in enumerate(zip(axes, programs)):
        data = data_dict[program][variable]
        data_cf = data_dict[program][variable_cf]

        if not same_x_scale:
            min_value = min(data.min(), data_cf.min())
            max_value = max(data.max(), data_cf.max())

        bins = np.linspace(min_value, max_value, bin_count)
        if x_scale == "log":
            bins = np.logspace(np.log10(min_value), np.log10(max_value), bin_count)
        elif x_scale == "symlog":
            bins = np.geomspace(max(min_value, 1e-3), max_value, bin_count)

        ax.set_xscale(x_scale)

        ax.hist(
            data,
            bins=bins,
            alpha=alpha,
            color=program_colors[program],
            label="QARYs with program",
        )
        ax.hist(
            data_cf,
            bins=bins,
            alpha=alpha_cf,
            color=program_colors[program],
            label="QARYs without program",
        )

        if idx == len(programs) - 1:
            ax.set_xlabel(help.prettify_label(variable))
        else:
            ax.set_xlabel("")

        ax.set_yticks([])

        pretty_label = help.prettify_label(program)
        split_pretty_label = help.split_title(pretty_label, 20)
        ax.set_ylabel(split_pretty_label, fontsize=13, rotation=90, labelpad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if not single_legend or idx == len(programs) - 1:
            ax.legend(loc=legend_locations[idx])

        # Set x-axis ticks and limits
        if x_scale == "symlog":
            threshold = symlog_threshold
            lower_exponent = int(np.floor(np.log10(abs(max(min_value, threshold)))))
            upper_exponent = int(np.ceil(np.log10(abs(max_value))))

            tick_values = sorted(
                list(
                    -np.logspace(
                        upper_exponent,
                        lower_exponent,
                        abs(upper_exponent - lower_exponent) + 1,
                    )
                )
                + [0]
                + list(
                    np.logspace(
                        lower_exponent,
                        upper_exponent,
                        upper_exponent - lower_exponent + 1,
                    )
                )
            )

            ax.set_xticks(tick_values)
            ax.set_xlim(-(10**lower_exponent), 10**upper_exponent)

        if xlim_bottom is not None:
            ax.set_xlim(left=xlim_bottom)

        if idx == len(programs) - 1:
            ax.set_xlabel(help.prettify_label(variable))
            # Add the following line to set the x-axis ticks for the last subplot:
            ax.xaxis.set_ticks_position("bottom")
        else:
            ax.spines["bottom"].set_visible(False)
            ax.set_xlabel("")
            # Add the following line to remove the x-axis ticks for all but the last subplot:
            ax.set_xticks([])

    if title:  # Add this check to add title if it's provided
        plt.suptitle(title, fontsize=14, y=1)
    if subtitle:
        fig.text(
            0.5, 0.96, subtitle, ha="center", va="center", fontsize=12, style="italic"
        )

    fig.tight_layout()

    # Return figure
    return fig


"""
Paired ECDFs
"""


def plot_ecdf_difference(
    data, programs, program_colors, xlabel="qarys", ylabel="ECDF", xscale="log"
):
    """
    Plots ECDF differences between two scenarios for multiple programs.

    Args:
        data: A dictionary containing program data.
                    Keys should be program names, and values should be DataFrames containing 'qarys' and 'qarys_cf' columns.
        programs: A list of program names to be plotted.
        program_colors: A dictionary containing colors for each program.
        xlabel: A string representing the x-axis label. Default is 'qarys'.
        ylabel: A string representing the y-axis label. Default is 'ECDF'.
        xscale: A string representing the x-axis scale. Default is 'log'.

    Returns:
        fig: A matplotlib figure object.
    """

    fig, ax = plt.subplots()

    data = {key: data[key] for key in programs}
    program_colors = {key: program_colors[key] for key in programs}

    legend_patches = [
        Patch(color=color, label=help.prettify_label(label))
        for label, color in program_colors.items()
    ]

    for program, color in program_colors.items():
        group_data = data[program]

        # extract qarys and qarys_cf
        qarys = group_data["qarys"]
        qarys_cf = group_data["qarys_cf"]

        # create ecdfs
        qarys_sorted = np.sort(qarys)
        qarys_cf_sorted = np.sort(qarys_cf)
        qarys_ecdf = np.arange(len(qarys_sorted)) / len(qarys_sorted)
        qarys_cf_ecdf = np.arange(len(qarys_cf_sorted)) / len(qarys_cf_sorted)

        # plot ecdfs
        ax.plot(qarys_sorted, qarys_ecdf, drawstyle="steps", color=color, label=program)
        ax.plot(
            qarys_cf_sorted,
            qarys_cf_ecdf,
            drawstyle="steps",
            color=color,
            linestyle="--",
        )

        # fill between ecdfs
        ax.fill_betweenx(
            qarys_ecdf,
            np.maximum(qarys_sorted, qarys_cf_sorted.min()),
            qarys_cf_sorted,
            where=qarys_cf_sorted < qarys_sorted,
            alpha=0.3,
            color=color,
        )

    # add legend
    ax.set_xscale(xscale)
    ax.set_xlabel(help.prettify_label(xlabel))
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_patches, frameon=False)

    # Remove the box around the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Return figure
    return fig


"""
Stacked histograms
"""


def plot_stacked_histograms(
    data,
    data_cf,
    column_names,
    bin_count=200,
    color_data=None,
    color_data_cf=None,
    log_scale=False,
):
    """
    Plots stacked histograms for multiple columns.

    Args:
        data: A list of data columns.
        data_cf: A list of counterfactual data columns.
        column_names: A list of column names.
        bin_count: An integer representing the number of bins. Default is 200.
        color_data: A list of colors for data columns. Default is None.
        color_data_cf: A list of colors for counterfactual data columns. Default is None.
        log_scale: A boolean representing whether to use log scale. Default is False.

    Returns:
        fig: A matplotlib figure object.
    """
    fig, ax = plt.subplots()

    max_value = max([col.max() for col in data + data_cf])
    min_value = min([col.min() for col in data + data_cf])

    bins = np.linspace(min_value, max_value, bin_count)
    if log_scale:
        bins = np.logspace(np.log10(min_value), np.log10(max_value), bin_count)
        ax.set_xscale("log")

    if color_data_cf is None:
        color_data_cf = ["#4B0082", "#800080", "#BA55D3"]

    if color_data is None:
        color_data = ["#008000", "#228B22", "#3CB371"]

    plt.hist(
        data_cf,
        bins=bins,
        stacked=True,
        alpha=0.1,
        color=color_data_cf,
        label=[f"{name} CF" for name in column_names],
    )
    plt.hist(
        data, bins=bins, stacked=True, alpha=0.5, color=color_data, label=column_names
    )

    plt.title("Stacked Histogram of QARYs by Attendee Type and CF Type")
    plt.xlabel("QARYs")
    plt.ylabel("Frequency")
    plt.legend(title="Attendee Type")

    # Return figure
    return fig


"""
Changing quality curves
"""


def plot_time_series_grid(
    data,
    variables,
    programs,
    participants,
    professions,
    program_colors,
    x_label="Years since program start",
    title=None,
    subtitle=None,
    title_y_coordinate=1.0,
    subtitle_y_coordinate=0.96,
    legend_location=None,
    apply_custom_padding=False,
    x_min=None,
    x_max=None,
    apply_symlog=False,
    symlog_linthresh=0.1,
    symlog_ymin_research_relevance=None,
    symlog_ymin_qarys=None,
    symlog_yticks_research_relevance=None,
    symlog_yticks_qarys=None,
    legend_horizontal_anchor=0.98,
):
    """
    Plots a grid of time series plots for multiple variables.

    Args:
        data: A dictionary of DataFrames, where the keys are program names and the values are DataFrames.
        variables: A list of variables to plot.
        programs: A list of programs to plot.
        participants: A list of participants to plot.
        professions: A list of professions to plot.
        program_colors: A dictionary of program colors, where the keys are program names and the values are colors.
        x_label: A string representing the x-axis label. Default is "Years since program start".
        title: A string representing the title. Default is None.
        subtitle: A string representing the subtitle. Default is None.
        title_y_coordinate: A float representing the y-coordinate of the title. Default is 1.0.
        subtitle_y_coordinate: A float representing the y-coordinate of the subtitle. Default is 0.96.
        legend_location: A string representing the legend location. Default is None.
        apply_custom_padding: A boolean representing whether to apply custom padding. Default is False.
        x_min: A float representing the minimum x value. Default is None.
        x_max: A float representing the maximum x value. Default is None.
        apply_symlog: A boolean representing whether to apply symlog scale. Default is False.
        symlog_linthresh: A float representing the linear threshold for symlog scale. Default is 0.1.
        symlog_ymin_research_relevance: A float representing the minimum y value for research relevance. Default is None.
        symlog_ymin_qarys: A float representing the minimum y value for QARYs. Default is None.
        symlog_yticks_research_relevance: A list of floats representing the y ticks for research relevance. Default is None.
        symlog_yticks_qarys: A list of floats representing the y ticks for QARYs. Default is None.
        legend_horizontal_anchor: A float representing the horizontal anchor for the legend. Default is 0.98.

    Returns:
        fig: A matplotlib figure.
    """
    custom_padding = {
        "research_relevance": 0.2,  # 10% padding for the problematic variable
        "qarys": 0.2,
    }
    default_padding = 0.05  # 5% padding for other variables

    fig, axes = plt.subplots(
        len(variables),
        len(professions),
        figsize=(3 * len(variables), 2 * len(professions)),
        sharex=True,
        sharey="row",
    )

    for variable_idx, variable in enumerate(variables):
        # Calculate the minimum and maximum y values for each variable
        min_y = np.inf
        max_y = -np.inf
        for profession_idx, profession in enumerate(professions):
            for program in programs:
                for participant in participants:
                    subset_data = data[
                        program
                    ]  # Get the DataFrame for the current program

                    if (
                        f"{variable}_over_t_{participant}_{profession}"
                        not in subset_data.columns
                    ):
                        continue

                    # Filter the data within x_min and x_max range
                    if x_min is not None:
                        subset_data = subset_data[subset_data["time_point"] >= x_min]
                    if x_max is not None:
                        subset_data = subset_data[subset_data["time_point"] <= x_max]

                    current_min_y = np.nanmin(
                        subset_data[f"{variable}_over_t_{participant}_{profession}"]
                    )
                    current_max_y = np.nanmax(
                        subset_data[f"{variable}_over_t_{participant}_{profession}"]
                    )
                    min_y = min(min_y, current_min_y)
                    max_y = max(max_y, current_max_y)

            if apply_custom_padding:
                padding = custom_padding.get(variable, default_padding)
            else:
                padding = default_padding

            y_range = max_y - min_y
            try:
                ax = axes[variable_idx, profession_idx]
            except:
                ax = axes[profession_idx]
            # ax.set_ylim(-0.3, max_y + padding * y_range)
            ax.spines[["top", "right"]].set_visible(False)  # Remove the border

            # Set x_min and x_max for each subplot if provided
            if x_min is not None:
                ax.set_xlim(left=x_min)
            if x_max is not None:
                ax.set_xlim(right=x_max)

            # Apply symlog scaling for 'research_relevance' and 'qarys' variables if the flag is set
            if apply_symlog and variable in ["research_relevance", "qarys"]:
                ax.set_yscale("symlog", linthresh=symlog_linthresh)

                def custom_symlog_tick_formatter(value, pos):
                    if np.isinf(value) or np.isnan(value) or value == 0:
                        return ""

                    exponent = np.floor(np.log10(np.abs(value)))
                    coeff = value / (10**exponent)

                    # We round the coefficient to one decimal place to avoid precision issues.
                    coeff = round(coeff, 1)

                    # If the coefficient is 1 or a multiple of 3, show the tick.
                    if coeff == 1 or coeff % 3 == 0:
                        # return '${:.0f}\\times10^{{{:.0f}}}$'.format(coeff, exponent)
                        return "{:.0f}".format(value)
                    else:
                        return ""

                # Apply the custom tick formatter
                # ax.yaxis.set_major_formatter(FuncFormatter(custom_symlog_tick_formatter))

                if (
                    variable == "research_relevance"
                    and symlog_yticks_research_relevance is not None
                ):
                    ax.set_yticks(symlog_yticks_research_relevance)
                    ax.set_ylim(bottom=symlog_ymin_research_relevance)
                elif variable == "qarys" and symlog_yticks_qarys is not None:
                    ax.set_yticks(symlog_yticks_qarys)
                    ax.set_ylim(bottom=symlog_ymin_qarys)

            # Apply the ylim top padding for the bottom row of subplots
            # if variable_idx == len(variables) - 1:
            #    ax.set_ylim(top=max_y + (padding + ylim_top_padding) * y_range)
            # else:
            #    ax.set_ylim(-0.3, max_y + padding * y_range)
            for program in programs:
                for participant_idx, participant in enumerate(participants):
                    if len(participants) == 1:
                        current_color = program_colors[program]
                    else:
                        current_color = program_colors[program][participant]
                    current_data = data[
                        program
                    ]  # Get the DataFrame for the current program

                    if (
                        f"{variable}_over_t_{participant}_{profession}"
                        not in current_data.columns
                    ):
                        continue

                    ax.plot(
                        current_data["time_point"],
                        current_data[f"{variable}_over_t_{participant}_{profession}"],
                        label=f"{program} - {participant}",
                        color=current_color,
                    )

                    # Plot the counterfactual with dashed line
                    ax.plot(
                        current_data["time_point"],
                        current_data[
                            f"{variable}_over_t_{participant}_{profession}_cf"
                        ],
                        color=current_color,
                        linestyle="--",
                    )

                    # Add shading between the actual and counterfactual curves
                    ax.fill_between(
                        current_data["time_point"],
                        current_data[f"{variable}_over_t_{participant}_{profession}"],
                        current_data[
                            f"{variable}_over_t_{participant}_{profession}_cf"
                        ],
                        color=current_color,
                        alpha=0.3,
                    )

            if variable_idx == 0:
                prettified_profession = help.prettify_label(profession)
                split_profession = help.split_title(prettified_profession, 20)
                ax.set_title(split_profession)

            if profession_idx == 0:
                if variable == "Probability of career end":
                    ax.set_ylabel("Probability of\nCareer End")
                else:
                    prettified_variable = help.prettify_label(variable)
                    split_variable = help.split_title(prettified_variable, 15)
                    ax.set_ylabel(split_variable)

            if (
                variable_idx == len(variables) - 1
            ):  # Only show x-axis labels for the bottom row of subplots
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel("")

        if variable == "research_relevance":
            ax.autoscale(enable=False, axis="y")
            ax.set_ylim(bottom=0 - padding, top=np.ceil(max_y + padding))
        elif variable == "qarys":
            ax.autoscale(enable=False, axis="y")
            ax.set_ylim(bottom=0 - padding, top=np.ceil(max_y + padding))
    #
    fig.subplots_adjust(hspace=0.4)

    # Create a list to store the legend patches
    legend_patches = []

    # Iterate over the programs and participant types
    for program in programs:
        for participant in participants:
            # Get the DataFrame for the current program
            current_data = data[program]

            # Check if the variable exists for the current program and participant
            column_example = f"{variables[0]}_over_t_{participant}_{professions[0]}"
            if column_example in current_data.columns:
                if len(participants) == 1:
                    legend_patches.append(
                        mpatches.Patch(
                            color=program_colors[program],
                            label=help.prettify_label(
                                program, capitalize_each_word=True
                            ),
                        )
                    )
                else:
                    legend_patches.append(
                        mpatches.Patch(
                            color=program_colors[program][participant],
                            label=help.prettify_label(
                                program, capitalize_each_word=True
                            )
                            + " - "
                            + help.prettify_label(participant),
                        )
                    )

    # Create a one-column legend outside the plots using bbox_to_anchor
    fig.legend(
        handles=legend_patches,
        loc=legend_location if legend_location else "lower center",
        bbox_to_anchor=(legend_horizontal_anchor, 0.43),
        ncol=1,
    )

    if title:  # Add this check to add title if it's provided
        plt.suptitle(title, fontsize=14, y=title_y_coordinate)
    if subtitle:
        fig.text(
            0.5,
            subtitle_y_coordinate,
            subtitle,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
        )

    # Return figure
    return fig


def plot_single_variable_time_series(
    data,
    variable,
    programs,
    participants,
    professions,
    program_colors,
    x_label="Years since program start",
    title=None,
    subtitle=None,
    title_y_coordinate=1.02,
    subtitle_y_coordinate=0.96,
    legend_location="lower center",
    legend_horizontal_anchor=0.98,
    ylim_padding=0.05,
    x_min=None,
    x_max=None,
    apply_symlog=False,
    symlog_linthresh=0.1,
    symlog_ymin=None,
    symlog_yticks=None,
):
    """
    This function creates a grid of time series plots for a single variable. Each row in the grid corresponds to a
    program and the columns correspond to professions.

    Args:
        data: A dictionary containing data for each program.
        variable: The variable to be plotted.
        programs: A list of programs to be plotted.
        participants: A list of participant types to be plotted.
        professions: A list of professions to be plotted.
        program_colors: A dictionary mapping programs and participants to colors.
        x_label: The label for the x-axis.
        title: The title for the entire figure.
        subtitle: The subtitle for the entire figure.
        title_y_coordinate: The y-coordinate for the title.
        subtitle_y_coordinate: The y-coordinate for the subtitle.
        legend_location: The location for the legend.
        legend_horizontal_anchor: The horizontal anchor for the legend.
        ylim_padding: The padding to be added to the y-axis limits.
        x_min: The minimum x-value.
        x_max: The maximum x-value.
        apply_symlog: A flag indicating whether to apply symlog scaling.
        symlog_linthresh: The linthresh parameter for symlog scaling.
        symlog_ymin: The minimum y-value for symlog scaling.
        symlog_yticks: The y-ticks for symlog scaling.

    Returns:
        fig: The figure object.
    """

    # Create a subplot for each program
    fig, axes = plt.subplots(
        len(programs),
        len(professions),
        figsize=(3 * len(programs), 2 * len(professions)),
        sharex=True,
        sharey="row",
    )

    # Loop over each program and profession
    for program_idx, program in enumerate(programs):
        for profession_idx, profession in enumerate(professions):
            # Set the current axes
            ax = axes[program_idx, profession_idx]

            # Initialize the min and max y values
            min_y = np.inf
            max_y = -np.inf

            # Loop over each participant
            for participant in participants:
                # Get the relevant data
                subset_data = data[program]

                # Check if the variable exists for the current program and participant
                if (
                    f"{variable}_over_t_{participant}_{profession}"
                    not in subset_data.columns
                ):
                    continue

                # Get the current min and max y values
                current_min_y = np.nanmin(
                    subset_data[f"{variable}_over_t_{participant}_{profession}"]
                )
                current_max_y = np.nanmax(
                    subset_data[f"{variable}_over_t_{participant}_{profession}"]
                )
                current_min_y_cf = np.nanmin(
                    subset_data[f"{variable}_over_t_{participant}_{profession}_cf"]
                )
                current_max_y_cf = np.nanmax(
                    subset_data[f"{variable}_over_t_{participant}_{profession}_cf"]
                )
                min_y = min(min_y, current_min_y)
                max_y = max(max_y, current_max_y)
                min_y = min(min_y, current_min_y_cf)
                max_y = max(max_y, current_max_y_cf)

                # Get the color for the current participant
                current_color = program_colors[program][participant]

                # Plot the data
                ax.plot(
                    subset_data["time_point"],
                    subset_data[f"{variable}_over_t_{participant}_{profession}"],
                    label=f"{program} - {participant}",
                    color=current_color,
                )

                # Plot the counterfactual with dashed line
                ax.plot(
                    subset_data["time_point"],
                    subset_data[f"{variable}_over_t_{participant}_{profession}_cf"],
                    color=current_color,
                    linestyle="--",
                )

                # Add shading between the actual and counterfactual curves
                ax.fill_between(
                    subset_data["time_point"],
                    subset_data[f"{variable}_over_t_{participant}_{profession}"],
                    subset_data[f"{variable}_over_t_{participant}_{profession}_cf"],
                    color=current_color,
                    alpha=0.3,
                )

                ax.spines[["top", "right"]].set_visible(False)  # Remove the border

            # Set the y-axis limits
            y_range = max_y - min_y
            ax.set_ylim(min_y - ylim_padding * y_range, max_y + ylim_padding * y_range)

            # Set x_min and x_max for each subplot if provided
            if x_min is not None:
                ax.set_xlim(left=x_min)
            if x_max is not None:
                ax.set_xlim(right=x_max)

            # Set the y-axis label
            if profession_idx == 0:
                prettified_y_label = help.prettify_label(
                    program, capitalize_each_word=True
                )
                split_y_label = help.split_title(prettified_y_label, max_length=15)
                ax.set_ylabel(split_y_label)
            else:
                ax.set_ylabel("")

            # Apply symlog scaling if applicable
            if apply_symlog:
                ax.set_yscale("symlog", linthresh=symlog_linthresh)
                if symlog_yticks is not None:
                    ax.set_yticks(symlog_yticks)
                    ax.set_ylim(bottom=symlog_ymin)

            # Set the x-axis label for the bottom row of subplots
            if program_idx == len(programs) - 1:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel("")

            if program_idx == 0:
                ax.set_title(help.prettify_label(profession))

    # Create a list to store the legend patches
    legend_patches = []

    # Iterate over the programs and participant types
    for program in programs:
        for participant in participants:
            # Get the DataFrame for the current program
            current_data = data[program]

            # Check if the variable exists for the current program and participant
            column_example = f"{variable}_over_t_{participant}_{professions[0]}"
            if column_example in current_data.columns:
                legend_patches.append(
                    mpatches.Patch(
                        color=program_colors[program][participant],
                        label=help.prettify_label(program, capitalize_each_word=True)
                        + " - "
                        + help.prettify_label(participant),
                    )
                )

    # Create a one-column legend outside the plots using bbox_to_anchor
    fig.legend(
        handles=legend_patches,
        loc=legend_location,
        bbox_to_anchor=(legend_horizontal_anchor, 0.43),
        ncol=1,
    )

    # Set the title and subtitle if they are provided
    if title:
        plt.suptitle(title, fontsize=14, y=title_y_coordinate)
    if subtitle:
        fig.text(
            0.5,
            subtitle_y_coordinate,
            subtitle,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
        )

    fig.subplots_adjust(hspace=0.4)

    # Return the figure
    return fig
