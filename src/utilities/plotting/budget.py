"""
Purpose: functions for plotting outcomes as a function of budget
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Common python packages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches  # for legend

# Helper functions
import utilities.plotting.helper_functions as help


"""
Functions
"""


def plot_budget_qarys(
    df,
    program_colors=None,
    title=None,
    xlabel=None,
    ylabel1=None,
    ylabel2=None,
    legend_pos="upper left",
    use_pseudo_log_x_scale=False,
    use_pseudo_log_y_scale=False,
    pseudolog_linthresh=1e0,
):
    """
    Plot the mean QARYS difference and cost-effectiveness as a function of budget, by program.

    Args:
        df (pd.DataFrame): Dataframe containing the results of the budget experiment.
        program_colors (dict): Dictionary mapping program names to colors.
        title (str): Title of the plot.
        xlabel (str): Label of the x-axis.
        ylabel1 (str): Label of the y-axis for the mean QARYS difference.
        ylabel2 (str): Label of the y-axis for the cost-effectiveness.
        legend_pos (str): Position of the legend.
        use_pseudo_log_x_scale (bool): Whether to use a pseudo-log scale for the x-axis.
        use_pseudo_log_y_scale (bool): Whether to use a pseudo-log scale for the y-axis.
        pseudolog_linthresh (float): Threshold for the pseudo-log scale.

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot.
    """
    programs = df["program"].unique()

    if program_colors is None:
        program_colors = plt.cm.get_cmap("tab10", len(programs))
        get_color = lambda program: program_colors(program)
    else:
        get_color = lambda program: program_colors[program]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3], "hspace": 0.1},
    )

    legend_entries = []

    for i, program in enumerate(programs):
        program_df = df[df["program"] == program]
        ax1.plot(
            program_df["target_budget"],
            program_df["mean_qarys_diff"],
            label=f"{program}",
            color=get_color(program),
        )
        legend_entries.append(f"{program}")

    ax1.set_ylabel(ylabel1 if ylabel1 else "Mean QARYS Difference")

    for i, program in enumerate(programs):
        program_df = df[df["program"] == program]
        ax2.plot(
            program_df["target_budget"],
            program_df["cost_effectiveness"],
            color=get_color(program),
        )

    ax2.set_ylabel(ylabel2 if ylabel2 else "Cost-Effectiveness")

    ax1.set_title(
        title if title else "Mean QARYS Difference & Cost-Effectiveness vs Budget"
    )

    ax1.ticklabel_format(useOffset=False, style="plain")
    ax2.ticklabel_format(useOffset=False, style="plain")

    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    if use_pseudo_log_x_scale:
        ax1.set_xscale("symlog", linthresh=pseudolog_linthresh)
        ax2.set_xscale("symlog", linthresh=pseudolog_linthresh)

    if use_pseudo_log_y_scale:
        ax1.set_yscale("symlog", linthresh=pseudolog_linthresh)
        ax2.set_yscale("symlog", linthresh=pseudolog_linthresh)

    # Create a shared legend
    legend_patches = [
        mpatches.Patch(
            color=get_color(program),
            label=help.prettify_label(program, capitalize_each_word=True),
        )
        for program in legend_entries
    ]
    ax1.legend(handles=legend_patches, loc=legend_pos)

    ax1.grid(alpha=0.3, which="both")
    ax2.grid(alpha=0.3, which="both")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    ax2.set_xlabel(xlabel if xlabel else "Budget")

    plt.tight_layout()

    # Return figure
    return fig


def plot_budget_n_participant(
    df,
    program_colors=None,
    participant_types=["n_participant", "n_attendee", "n_contender"],
    title=None,
    xlabel=None,
    legend_pos="upper left",
    use_pseudo_log_scale=False,
    force_zero_ylim=True,
    x_max_attendee=None,
):
    """
    Plots average number of participants as a function of target budget, for each program.

    Args:
        df (pd.DataFrame): Dataframe containing the data to plot.
        program_colors (dict or None): Dictionary mapping program names to colors.
        participant_types (list): List of participant types to plot.
        title (str or None): Title of the plot.
        xlabel (str or None): Label for the x-axis.
        legend_pos (str): Position of the legend.
        use_pseudo_log_scale (bool): Whether to use a pseudo-log scale for the y-axis.
        force_zero_ylim (bool): Whether to force the y-axis to start at 0.
        x_max_attendee (int or None): Maximum number of attendees to plot.

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot.
    """
    programs = df["program"].unique()

    if program_colors is None:
        program_colors = plt.cm.get_cmap("tab10", len(programs))
        get_color = lambda program: program_colors(program)
    else:
        get_color = lambda program: program_colors[program]

    fig, axes = plt.subplots(
        len(participant_types),
        1,
        figsize=(8, 8 * len(participant_types)),
        sharex=False,
        gridspec_kw={"hspace": 0.3},
    )

    legend_entries = []

    for i, ptype in enumerate(participant_types):
        if len(participant_types) == 1:
            ax = axes
        else:
            ax = axes[i]
        ptype_string = "n_" + ptype
        for j, program in enumerate(programs):
            program_df = df[df["program"] == program]
            if ptype_string in program_df.columns:
                ax.plot(
                    program_df["target_budget"],
                    program_df[ptype_string],
                    label=f"{program}",
                    color=get_color(program),
                )
                legend_entries.append(f"{program}")

        ax.set_ylabel("Mean number of " + ptype + "s")

        # Set the ylim bottom to 0 if force_zero_ylim is True
        if force_zero_ylim:
            ax.set_ylim(bottom=0)

        # Cap the x_max of the x-axis for attendee subplot only
        if x_max_attendee is not None and ptype == "attendee":
            ax.set_xlim(left=0, right=x_max_attendee)

            # Adjust y-axis to go from 0 to the maximum over the range of the x-axis
            ymax = df[df["target_budget"] <= x_max_attendee][ptype_string].max()
            ymax_with_buffer = ymax * 1.1
            ax.set_ylim(top=ymax_with_buffer)

    if len(participant_types) > 1:
        axes[-1].set_xlabel(xlabel if xlabel else "Budget")

    if title:
        fig.suptitle(title, y=0.93)

    if use_pseudo_log_scale:
        for ax in axes:
            ax.set_yscale("symlog")

    # Create a shared legend
    legend_patches = [
        mpatches.Patch(
            color=get_color(program),
            label=help.prettify_label(program, capitalize_each_word=True),
        )
        for program in list(set(legend_entries))
    ]
    if len(participant_types) == 1:
        axes.legend(handles=legend_patches, loc=legend_pos)
    else:
        axes[0].legend(handles=legend_patches, loc=legend_pos)

    if len(participant_types) == 1:
        axes.grid(alpha=0.3, which="both")
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["left"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
    else:
        for ax in axes:
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.grid(alpha=0.3, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    return fig
