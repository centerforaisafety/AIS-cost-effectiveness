"""
Purpose: impact and ability plots (typically shared across program types)
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
import textwrap
import matplotlib.lines as lines
import numpy as np  # for sorting arrays

# For mean scientist-equivalent ability
import utilities.functions.quality_adjustment as qa


"""
Compare one variable from each program
"""


def extract_ability_parameters(
    default_parameters,
    professional_types=["undergrad", "phd"],
    participant_types=["student"],
):
    """
    This function extracts ability parameters from a dictionary of dictionaries.

    Args:
        default_parameters (dict): A dictionary of dictionaries containing program parameters.
        professional_types (list): A list of professional types to extract.
        participant_types (list): A list of participant types to extract.

    Returns:
        DataFrame: A pandas DataFrame with columns "program", "student type", "ability_at_first", "ability_at_pivot", and "ability_pivot_point".
    """

    # Initialize an empty list to store the data
    data = []

    # Loop through each program in the dictionary
    for program, params in default_parameters.items():
        for participant_type in participant_types:
            for professional_type in professional_types:
                person_type = f"{participant_type}_{professional_type}"
                mainline_params = params.params["mainline"]

                # Extract the ability parameters for the student type
                ability_at_first = mainline_params.get(
                    f"ability_at_first_{person_type}"
                )
                ability_at_pivot = mainline_params.get(
                    f"ability_at_pivot_{person_type}"
                )
                ability_pivot_point = mainline_params.get(
                    f"ability_pivot_point_{person_type}"
                )

                # If all parameters are not None, append them to the data list
                if (
                    ability_at_first is not None
                    and ability_at_pivot is not None
                    and ability_pivot_point is not None
                ):
                    data.append(
                        [
                            program,
                            participant_type,
                            professional_type,
                            ability_at_first,
                            ability_at_pivot,
                            ability_pivot_point,
                        ]
                    )

    # Convert the data list into a DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "program",
            "participant_type",
            "professional_type",
            "ability_at_first",
            "ability_at_pivot",
            "ability_pivot_point",
        ],
    )

    return df


def plot_mean_ability_piecewise(
    df,
    program_colors,
    alpha=0.3,
    line_alpha=0.7,
    line_width=2,
    horizontal_line_alpha=0.3,
    horizontal_line_style="--",
    title="Participant Ability Levels",
    legend_loc="upper right",
    dotted_asymptote=False,
):
    """
    This function plots the mean ability as a function of participation ordering for each program-participant_type pair.

    Args:
        df (DataFrame): A pandas DataFrame with columns "program", "student_type", "ability_at_first", "ability_at_pivot", and "ability_pivot_point".
        program_colors (dict): A dictionary mapping program names to colors.
        alpha (float): The alpha value for the grid.
        line_alpha (float): The alpha value for the lines.
        line_width (int): The line width.
        horizontal_line_alpha (float): The alpha value for the horizontal lines.
        horizontal_line_style (str): The line style for the horizontal lines.
        title (str): The title of the plot.
        legend_loc (str): The location of the legend.

    Returns:
        None
    """
    # Set up the plot
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle(title)
    axs[0].set_ylabel("Ability of the marginal student")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("Number of participants in program")
    axs[1].set_ylabel("Mean ability")
    axs[0].grid(alpha=alpha)
    axs[1].grid(alpha=alpha)

    # Loop through the rows in the DataFrame
    for _, row in df.iterrows():
        # Extract the parameters
        program = row["program"]
        participant_type = row["participant_type"]
        professional_type = row["professional_type"]
        ability_at_first = row["ability_at_first"]
        ability_at_pivot = row["ability_at_pivot"]
        ability_pivot_point = row["ability_pivot_point"]

        # Calculate abilities using the piecewise function
        n_samples_range = np.arange(1, 101)
        abilities = [
            qa.piecewise_function(
                x, ability_at_first, ability_at_pivot, ability_pivot_point
            )
            for x in n_samples_range
        ]
        mean_abilities = np.cumsum(abilities) / n_samples_range

        # Plot the abilities and mean abilities
        line_style = "--" if professional_type == "phd" else "-"
        axs[0].plot(
            n_samples_range,
            abilities,
            label=help.prettify_label(program, capitalize_each_word=True)
            + " - "
            + help.prettify_label(participant_type),
            alpha=line_alpha,
            linewidth=line_width,
            color=program_colors[program],
            linestyle=line_style,
        )
        axs[1].plot(
            n_samples_range,
            mean_abilities,
            alpha=line_alpha,
            linewidth=line_width,
            color=program_colors[program],
            linestyle=line_style,
        )

        # Plot the horizontal dotted line at ability_at_pivot
        if dotted_asymptote:
            axs[1].axhline(
                y=ability_at_pivot,
                color=program_colors[program],
                linestyle=horizontal_line_style,
                alpha=horizontal_line_alpha,
            )

    # Set y-axis scale and ticks
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlim(left=0)

    # Create a list to store the legend patches
    legend_patches = []

    # Iterate over the programs and professional types
    for program in program_colors.keys():
        for professional_type in ["undergrad", "phd"]:
            # Get the DataFrame for the current program
            current_data = df[
                (df["program"] == program)
                & (df["professional_type"] == professional_type)
            ]

            if len(current_data) == 1:
                legend_patches.append(
                    lines.Line2D(
                        [],
                        [],
                        color=program_colors[program],
                        lw=2,
                        label=f"{help.prettify_label(program, capitalize_each_word=True)}: {help.prettify_label(professional_type)}",
                        linestyle="--" if professional_type == "phd" else "-",
                    )
                )
            else:
                pass

    # Create a one-column legend outside the plots using bbox_to_anchor
    fig.legend(
        handles=legend_patches,
        loc=legend_loc,  # if legend_loc else "lower center",
        bbox_to_anchor=(0.9, 0.9),
        ncol=1,
    )

    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    # Show the plot
    return fig


"""
New impact and ability plots
"""


def even_split_label(label, max_width=8):
    wrapper = textwrap.TextWrapper(width=max_width, break_long_words=False)
    lines = wrapper.wrap(label)
    return "\n".join(lines)


def plot_impact_bar_chart(
    df,
    use_symlog_scale=True,
    ylim_bottom=None,
    colors=("green", "orange"),
    figsize=(12, 6),
    labels_on_bars=True,
):
    """
    Plot a bar chart of the impact of research avenues.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        use_symlog_scale (bool): Whether to use a symlog scale for the y-axis.
        ylim_bottom (float): The bottom of the y-axis.
        colors (tuple): The colors to use for the bars.
        figsize (tuple): The size of the figure.
        labels_on_bars (bool): Whether to show the labels on the bars.

    Returns:
        fig: The figure containing the plot.
    """
    # Create a copy of the DataFrame to avoid modifying the original one
    df = df.copy()

    # Extract research avenues and impact values
    df["With Compute"] = df["Research Avenue"].str.endswith("with_compute").astype(str)
    df["Research Avenue"] = (
        df["Research Avenue"]
        .str.replace("_with_compute", "")
        .str.replace("_without_compute", "")
        .str.replace("impact_", "")
    )
    df_pivot = (
        df.pivot_table(index="Research Avenue", columns="With Compute", values="Impact")
        .fillna(0)
        .reset_index()
    )

    # Reorder the research avenues as requested
    custom_order = sorted(
        df_pivot["Research Avenue"].tolist(),
        key=lambda x: (x == "dangerous_capabilities", x == "nothing", x),
    )
    df_pivot = df_pivot.set_index("Research Avenue").loc[custom_order].reset_index()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df_pivot["Research Avenue"]))
    bar_width = 0.3
    opacity = 0.8
    bars_with_compute = ax.bar(
        x - bar_width / 2,
        df_pivot.loc[:, "True"],
        bar_width,
        alpha=opacity,
        color=colors[0],
        label="With Compute",
    )
    bars_without_compute = ax.bar(
        x + bar_width / 2,
        df_pivot.loc[:, "False"],
        bar_width,
        alpha=opacity,
        color=colors[1],
        label="Without Compute",
    )

    # Configure the plot
    ax.set_xticks(x)
    if not labels_on_bars:
        ax.set_xticklabels(
            [
                even_split_label(help.prettify_label(label))
                for label in df_pivot["Research Avenue"]
            ],
            rotation=0,
            ha="center",
        )
    else:
        ax.set_xticklabels([])
        ax.set_xticks([])  # Remove x-axis ticks
        ax.spines["bottom"].set_visible(False)  # Remove x-axis line

    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    if use_symlog_scale:
        ax.set_yscale("symlog", linthresh=0.01)
        ax.yaxis.grid(alpha=0.3)

    if labels_on_bars:
        for i, (bar_with, bar_without) in enumerate(
            zip(bars_with_compute, bars_without_compute)
        ):
            label = even_split_label(
                help.prettify_label(df_pivot["Research Avenue"][i])
            )
            height_with = bar_with.get_height()
            height_without = bar_without.get_height()

            if height_with == 0:
                max_height = height_without
                max_bar = bar_without
            elif height_without == 0:
                max_height = height_with
                max_bar = bar_with
            else:
                max_height = max(abs(height_with), abs(height_without))
                max_bar = bar_with if abs(height_with) == max_height else bar_without

            label_offset_x = 6.5 if max_height >= 0 else 0
            label_offset_y = 3 if max_height >= 0 else -3
            label_va = "bottom" if max_height >= 0 else "top"

            ax.annotate(
                label,
                xy=(max_bar.get_x() + max_bar.get_width() / 2, max_height),
                xytext=(label_offset_x, label_offset_y),
                textcoords="offset points",
                ha="center",
                va=label_va,
            )

    ax.legend(loc="lower left")

    # Return figure
    return fig
