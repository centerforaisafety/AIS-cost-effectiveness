"""
Purpose: plot robustness results
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
import matplotlib.patches as mpatches  # for legend
import numpy as np  # for sorting arrays
import copy  # for deep-copying objects
import inspect

# Simulations
from squigglepy.numbers import K, M


def robustness_analysis(
    programs,
    default_parameters,
    master_functions,
    parameter_name,
    parameter_values,
    outcome_var="qarys",
    n_sim=1000,
):
    """
    Perform a robustness analysis for a set of programs.

    Args:
        programs (list): A list of program names.
        default_parameters (dict): A dictionary where keys are program names and values are default parameters for each program.
        master_functions (dict): A dictionary where keys are program names and values are master functions for each program.
        parameter_name (str): The name of the parameter to vary.
        parameter_values (list): A list of values to vary the parameter by.
        outcome_var (str, optional): The name of the outcome variable to plot. Default is "qarys".
        n_sim (int, optional): The number of simulations to run for each program. Default is 1000.

    Returns:
        df (pandas.DataFrame): A dataframe containing the results of the robustness analysis.
    """
    df = pd.DataFrame(
        columns=[
            "program",
            "parameter_name",
            "parameter_value",
            "parameter_value_default",
            "outcome_value",
            "outcome_value_default",
        ]
    )

    if not isinstance(programs, list):
        programs = [programs]

    for i in programs:
        params = default_parameters[i].params["mainline"]
        params_cf = default_parameters[i].params["mainline_cf"]

        master_function = master_functions[i].mfn

        default = params[parameter_name]
        default_cf = params_cf[parameter_name]

        (
            params,
            params_sampled,
            derived_params_sampled,
            derived_functions,
        ) = master_function(params, params_cf, n_sim)

        default_qarys = np.mean(derived_params_sampled[outcome_var])
        default_qarys_cf = np.mean(derived_params_sampled[f"{outcome_var}_cf"])
        default_qarys_diff = default_qarys - default_qarys_cf
        default_cost_effectiveness = default_qarys_diff / (
            params["target_budget"] / (1 * M)
        )

        for item in parameter_values:
            print(f"Computing program {i} with {parameter_name} = {item}...")
            params[parameter_name] = item  # Update the parameter
            params_cf[
                parameter_name
            ] = item  # Update the parameter in the counterfactual

            (
                params,
                params_sampled,
                derived_params_sampled,
                derived_functions,
            ) = master_function(params, params_cf, n_sim)

            qarys = np.mean(derived_params_sampled[outcome_var])
            qarys_cf = np.mean(derived_params_sampled[f"{outcome_var}_cf"])
            qarys_diff = qarys - qarys_cf
            cost_effectiveness = qarys_diff / (params["target_budget"] / (1 * M))

            new_row = pd.DataFrame(
                {
                    "program": [i],
                    "parameter_name": [parameter_name],
                    "parameter_value": [item],
                    "parameter_value_default": [default],
                    "outcome_value": [cost_effectiveness],
                    "outcome_value_default": [default_cost_effectiveness],
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)

    return df


"""
Helper function
"""


def custom_deep_copy_dict(original_dict):
    """
    Deep copy a dictionary.

    Args:
        original_dict (dict): The dictionary to deep copy.

    Returns:
        new_dict (dict): The deep copy of the dictionary.
    """
    new_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, dict):
            new_dict[key] = custom_deep_copy_dict(value)
        elif not inspect.ismodule(value):  # Ignore module objects explicitly
            new_dict[key] = copy.deepcopy(value)
        else:
            new_dict[key] = value
    return new_dict


"""
Plot
"""


def process_robustness_data(
    programs,
    robustness_changes,
    default_parameters,
    master_functions,
    n_sim,
):
    """
    Process the robustness data for plotting.

    Args:
        programs (list): A list of program names.
        robustness_changes (dict): A dictionary where keys are parameter names and values are lists of values to vary the parameter by.
        default_parameters (dict): A dictionary where keys are program names and values are default parameters for each program.
        master_functions (dict): A dictionary where keys are program names and values are master functions for each program.
        n_sim (int): The number of simulations to run for each program.

    Returns:
        robustness_data (dict): A dictionary where keys are parameter names and values are dataframes containing the results of the robustness analysis.
    """
    robustness_data = {}

    for key, value in robustness_changes.items():
        copy_default_parameters = custom_deep_copy_dict(default_parameters)

        df = robustness_analysis(
            programs,
            copy_default_parameters,
            master_functions,
            key,
            value,
            n_sim=n_sim,
        )

        robustness_data[key] = df

    return robustness_data


def plot_robustness_multiple(
    robustness_data,
    program_colors,
    y_label="Cost-effectiveness\n(expected QARYs per $1M)",
    title="Robustness plot",
    alpha_robustness_default=1,
    alpha_robustness_points=0.5,
    alpha_robustness_lines=0.1,
    y_log_scale=True,
    global_ylim=False,
    global_ylim_bottom=None,
    axis_text_skip=1,
    xlim_buffer=0.3,
    ylim_buffer=0.3,
    linthresh=1e0,
    show_gridlines=False,
):
    """
    Plot the results of a robustness analysis.

    Args:
        robustness_data (dict): A dictionary where keys are parameter names and values are dataframes containing the results of the robustness analysis.
        program_colors (dict): A dictionary where keys are program names and values are colors to use for each program.
        y_label (str): The y-axis label.
        title (str): The title of the plot.
        alpha_robustness_default (float): The alpha value to use for the default parameter value.
        alpha_robustness_points (float): The alpha value to use for the robustness points.
        alpha_robustness_lines (float): The alpha value to use for the robustness lines.
        y_log_scale (bool): Whether to use a log scale for the y-axis.
        global_ylim (bool): Whether to use the same y-axis limits for all plots.
        global_ylim_bottom (float): The bottom y-axis limit to use for all plots.
        axis_text_skip (int): The number of ticks to skip on the x-axis.
        xlim_buffer (float): The buffer to use for the x-axis limits.
        ylim_buffer (float): The buffer to use for the y-axis limits.
        linthresh (float): The threshold to use for the log scale.
        show_gridlines (bool): Whether to show gridlines.

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
    """
    num_variables = len(robustness_data.keys())
    nrows = num_variables // 2 if num_variables % 2 == 0 else (num_variables // 2) + 1
    ncols = 2 if num_variables > 1 else 1

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(8 * (ncols**0.5), 8 * (nrows**0.5))
    )
    if nrows * ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)  # Remove the border

    ymin, ymax = float("inf"), float("-inf")

    for idx, (key, df) in enumerate(robustness_data.items()):
        ax = axes[idx]

        if global_ylim:
            ymin = min(
                ymin, df["outcome_value_default"].min(), df["outcome_value"].min()
            )
            ymax = max(
                ymax, df["outcome_value_default"].max(), df["outcome_value"].max()
            )

        proxy_artists = {}
        for program in df["program"].unique():
            pretty_label = help.prettify_label(program, capitalize_each_word=True)
            proxy_artists[program] = mpatches.Patch(
                color=program_colors[program], alpha=1, label=pretty_label
            )
            program_df = df[df["program"] == program]

            for index, row in program_df.iterrows():
                param_default = row["parameter_value_default"]
                param = row["parameter_value"]
                outcome_default = row["outcome_value_default"]
                outcome = row["outcome_value"]

                ax.scatter(
                    param_default,
                    outcome_default,
                    marker="o",
                    label=row["program"],
                    color=program_colors[program],
                    alpha=alpha_robustness_default - alpha_robustness_points,
                )
                ax.scatter(
                    param,
                    outcome,
                    marker="o",
                    color=program_colors[program],
                    alpha=alpha_robustness_points,
                )

                ax.plot(
                    [param_default, param],
                    [outcome_default, outcome],
                    linestyle="--",
                    color=program_colors[program],
                    alpha=alpha_robustness_lines,
                )

        ax.ticklabel_format(style="plain", axis="both")

        if y_log_scale:
            ymin = min(df["outcome_value_default"].min(), df["outcome_value"].min())
            ymax = max(df["outcome_value_default"].max(), df["outcome_value"].max())

            yticks = help.generate_yticks(ymin, ymax)
            ax.set_yticks(yticks)

            # Determine whether to use symlog or log scale
            contains_negative = any(df["outcome_value_default"] < 0) or any(
                df["outcome_value"] < 0
            )
            if contains_negative:
                print("Outcome contains negative value, using symlog scale")
                scale_type = "symlog"
                ax.set_yscale(scale_type, linthresh=linthresh)
            else:
                scale_type = "log"
                ax.set_yscale(scale_type)

            # Hide every nth y-axis tick label
            for index, label in enumerate(ax.yaxis.get_ticklabels()):
                if index % axis_text_skip != 0:
                    label.set_visible(False)

        # Add horizontal gridlines to the plot
        if show_gridlines:
            ax.yaxis.grid(
                True,  # Turns on the gridlines
                linestyle="--",  # Sets the gridlines to be dashed
                color="grey",  # Sets the gridlines color to grey
                alpha=0.1,  # Sets the gridlines to be very low-alpha
                which="both",  # Sets gridlines at both major and minor ticks
            )

        pretty_key = help.prettify_label(key)
        ax.set_xlabel(f"{pretty_key}")
        ax.set_ylabel(y_label)
        if title not in ["", None]:
            ax.set_title(f"{pretty_key}: {title}")

        xmin = min(df["parameter_value_default"].min(), df["parameter_value"].min())
        xmax = max(df["parameter_value_default"].max(), df["parameter_value"].max())

        if xmin < 0:
            xmin = xmin * (1 + ylim_buffer)
        elif ymax < 0:
            xmax = xmax * (1 - ylim_buffer)
        else:
            xmin = xmin * (1 - ylim_buffer)
            xmax = xmax * (1 + ylim_buffer)

        handles = [proxy_artists[program] for program in df["program"].unique()]
        ax.legend(handles=handles)

    if global_ylim:
        for ax in axes:
            # If global_ylim_bottom is not None, use it as the minimum y value
            if global_ylim_bottom is not None:
                ax.set_ylim(global_ylim_bottom, ymax * (1 + ylim_buffer))
            else:
                ax.set_ylim(ymin * (1 - ylim_buffer), ymax * (1 + ylim_buffer))

    if nrows * ncols > len(robustness_data):
        axes[-1].axis("off")

    plt.tight_layout()

    # Return figure
    return fig


"""
Compare across scenarios
"""


# Process the data into a new DataFrame
def transform_data_robustness_to_df(data_robustness):
    """
    Transforms the dictionary of dataframes from `get_scenario_data` function.

    Args:
        data_robustness (dict): Dictionary of dataframes from `get_scenario_data` function.

    Returns:
        pd.DataFrame: Dataframe with the following columns:
    """
    data = []
    for program in data_robustness.keys():
        program_dict = data_robustness[program]
        budget = program_dict["mainline"]["target_budget"][0]
        for scenario in program_dict.keys():
            scenario_df = program_dict[scenario]
            qarys_diff = np.mean(scenario_df["qarys"]) - np.mean(
                scenario_df["qarys_cf"]
            )
            data.append(
                {
                    "program": program,
                    "scenario": scenario,
                    "qarys_diff": qarys_diff,
                    "target_budget": budget,
                }
            )
    return pd.DataFrame(data)


def process_data_for_scenario_robustness_plot(df):
    """
    Process the data for the scenario robustness plot.

    Args:
        df (pd.DataFrame): Dataframe with the following columns:
            - program (str): Program name
            - scenario (str): Scenario name
            - qarys_diff (float): Difference in QARYs between the mainline and counterfactual scenarios
            - target_budget (float): Target budget for the program

    Returns:
        pd.DataFrame: Dataframe with the following columns:
            - program (str): Program name
            - scenario (str): Scenario name
            - qarys_diff (float): Difference in QARYs between the mainline and counterfactual scenarios
            - target_budget (float): Target budget for the program
            - cost_effectiveness (float): Cost-effectiveness of the program
    """
    df["cost_effectiveness"] = df["qarys_diff"] / (df["target_budget"] / (1 * M))
    all_scenarios = df["scenario"].unique()

    # Add rows with NaN values for missing program-scenario pairs
    for program in df["program"].unique():
        program_scenarios = df[df["program"] == program]["scenario"].unique()
        for scenario in all_scenarios:
            # Skip the mainline scenario and scenarios not present in the program
            if scenario == "mainline" or scenario not in program_scenarios:
                continue

            if not ((df["program"] == program) & (df["scenario"] == scenario)).any():
                df = df.append(
                    {
                        "program": program,
                        "scenario": scenario,
                        "qarys_diff": np.nan,
                        "target_budget": np.nan,
                        "cost_effectiveness": np.nan,
                    },
                    ignore_index=True,
                )

    # Sort the DataFrame based on the order of scenarios in the scenario_groups
    scenario_order = {scenario: i for i, scenario in enumerate(all_scenarios)}
    df["scenario_order"] = df["scenario"].map(scenario_order)
    df.sort_values(["program", "scenario_order"], inplace=True)
    df.drop("scenario_order", axis=1, inplace=True)

    return df


def plot_benefit_cost_effectiveness_robustness_charts(
    df, program_colors, legend_location=None
):
    """
    Plots the benefit and cost effectiveness of programs across multiple robustness scenarios.

    Args:
        df (pd.DataFrame): Dataframe.
        program_colors (dict): Dictionary mapping program names to colors.
        legend_location (str): Location of the legend.

    Returns:
        fig (matplotlib.figure.Figure): Figure.
    """

    def plot_chart(ax, column, title, log_scale=True, xtick_labels=None):
        """
        Plots a bar chart.
        """
        for i, (program, program_df) in enumerate(df.groupby("program")):
            print(len(df["program"].unique()))
            bar_width = 1 / (len(df["program"].unique()) + 1)
            positions = (
                np.arange(len(df["scenario"].unique()))
                + i * bar_width
                - (len(df["program"].unique()) - 3) / (len(df["program"].unique()) * 2)
            )

            for pos, scenario in zip(positions, df["scenario"].unique()):
                data_series = program_df.loc[program_df["scenario"] == scenario, column]
                if not data_series.empty:
                    data = data_series.values[0]
                    ax.bar(
                        pos,
                        data,
                        bar_width,
                        color=program_colors[program],
                        label=program,
                    )

        ax.set_xticks(np.arange(len(df["scenario"].unique())) + bar_width)
        ax.set_ylabel(title)

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$")
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.grid(alpha=0.3, axis="y", linestyle="--", which="both")
        ax.xaxis.grid(False)

        if xtick_labels:
            ax.set_xticklabels(
                xtick_labels, rotation=60, ha="right", rotation_mode="anchor"
            )
        else:
            ax.set_xticks([])

    def insert_newline_middle(label, index_shift=1):
        """
        Inserts a newline in the middle of a string.
        """
        middle_index = len(label) // 2
        nearest_index = label.find("_", middle_index - index_shift)
        if nearest_index == -1:
            return label
        return label[:nearest_index] + "\n" + label[nearest_index + 1 :]

    df = process_data_for_scenario_robustness_plot(df)

    x_labels = [
        # help.prettify_label(help.split_title(scenario, max_length=10))
        help.prettify_label(insert_newline_middle(scenario, index_shift=2))
        for scenario in df["scenario"].unique()
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    plot_chart(
        ax1, "qarys_diff", "Benefit\n(counterfactual expected QARYs)", xtick_labels=None
    )
    plot_chart(
        ax2,
        "cost_effectiveness",
        "Cost-effectiveness\n(expected QARYs per $1M)",
        xtick_labels=x_labels,
    )

    # Create the legend
    handles, labels = ax1.get_legend_handles_labels()
    pretty_labels = [
        help.prettify_label(label, capitalize_each_word=True) for label in labels
    ]
    by_label = dict(zip(pretty_labels, handles))
    if legend_location is None:
        legend_location = (0.98, 0.5)
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=legend_location,
    )

    plt.tight_layout()
    return fig
