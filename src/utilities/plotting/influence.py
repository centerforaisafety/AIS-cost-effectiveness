"""
Purpose: plots that show what you would need to believe about the chances of 
    influencing research avenues to achieve some level of benefit
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Sampling
import utilities.sampling.simulate_results as so

# Helper functions
import utilities.plotting.helper_functions as help

# Common python packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# Simulations
from squigglepy.numbers import K, M


"""
Functions
"""


def simulate_benefit_under_differing_influence(
    default_params,
    p_influence_values,
    programs,
    research_relevances,
    master_functions,
    n_sim,
    time_points,
):
    """
    Simulate the benefits of different programs under varying influence.

    Args:
        default_params (dict): Dictionary of parameters for each program.
        p_influence_values (list): List of influence values to loop over.
        programs (list): List of programs to loop over.
        research_relevances (dict): Dictionary of research relevance baselines and endlines.
        master_functions (dict): Dictionary of master functions.
        n_sim (int): Number of simulations to run.
        time_points (int): Number of time points to evaluate.

    Returns:
        results_df (pd.DataFrame): A DataFrame containing the simulation results.
    """
    influence_params = {}
    results = []

    for p_influence in p_influence_values:
        for program in programs:
            program_params = default_params[program].params.copy()

            participant_contender = program_params["mainline"].get(
                "participant_contender", False
            )
            participant_attendee = program_params["mainline"].get(
                "participant_attendee", False
            )
            participant_undergrad = program_params["mainline"].get(
                "years_until_phd_undergrad", False
            )
            participant_phd = program_params["mainline"].get("student_phd", False)

            if participant_contender:
                for researcher_type in ["scientist", "professor", "engineer", "phd"]:
                    baseline_key = (
                        f"research_relevance_contender_{researcher_type}_baseline"
                    )
                    endline_key = (
                        f"research_relevance_contender_{researcher_type}_endline"
                    )
                    after_program_key = (
                        f"research_relevance_contender_{researcher_type}"
                    )
                    during_program_key = (
                        f"research_relevance_during_program_contender_{researcher_type}"
                    )

                    program_params["mainline"][after_program_key] = (
                        research_relevances[program][baseline_key] * (1 - p_influence)
                        + research_relevances[program][endline_key] * p_influence
                    )
                    program_params["mainline"][
                        during_program_key
                    ] = research_relevances[program][endline_key]

                    program_params["mainline_cf"][
                        after_program_key
                    ] = research_relevances[program][baseline_key]
                    program_params["mainline_cf"][
                        during_program_key
                    ] = research_relevances[program][baseline_key]

            if participant_attendee:
                for researcher_type in ["scientist", "professor", "engineer", "phd"]:
                    baseline_key = (
                        f"research_relevance_attendee_{researcher_type}_baseline"
                    )
                    endline_key = (
                        f"research_relevance_attendee_{researcher_type}_endline"
                    )
                    after_program_key = f"research_relevance_attendee_{researcher_type}"
                    during_program_key = (
                        f"research_relevance_during_program_attendee_{researcher_type}"
                    )

                    program_params["mainline"][after_program_key] = (
                        research_relevances[program][baseline_key] * (1 - p_influence)
                        + research_relevances[program][endline_key] * p_influence
                    )
                    program_params["mainline"][
                        during_program_key
                    ] = research_relevances[program][endline_key]

                    program_params["mainline_cf"][
                        after_program_key
                    ] = research_relevances[program][baseline_key]
                    program_params["mainline_cf"][
                        during_program_key
                    ] = research_relevances[program][baseline_key]

            if participant_undergrad:
                for researcher_type in ["undergrad_via_phd", "undergrad_not_via_phd"]:
                    baseline_key = (
                        f"research_relevance_student_{researcher_type}_baseline"
                    )
                    endline_key = (
                        f"research_relevance_student_{researcher_type}_endline"
                    )
                    after_program_key = f"research_relevance_student_{researcher_type}"
                    during_program_key = (
                        f"research_relevance_during_program_student_{researcher_type}"
                    )

                    program_params["mainline"][after_program_key] = (
                        research_relevances[program][baseline_key] * (1 - p_influence)
                        + research_relevances[program][endline_key] * p_influence
                    )
                    program_params["mainline"][
                        during_program_key
                    ] = research_relevances[program][endline_key]

                    program_params["mainline_cf"][
                        after_program_key
                    ] = research_relevances[program][baseline_key]
                    program_params["mainline_cf"][
                        during_program_key
                    ] = research_relevances[program][baseline_key]

            if participant_phd:
                for researcher_type in ["phd"]:
                    baseline_key = (
                        f"research_relevance_student_{researcher_type}_baseline"
                    )
                    endline_key = (
                        f"research_relevance_student_{researcher_type}_endline"
                    )
                    after_program_key = f"research_relevance_student_{researcher_type}"
                    during_program_key = (
                        f"research_relevance_during_program_student_{researcher_type}"
                    )

                    program_params["mainline"][after_program_key] = (
                        research_relevances[program][baseline_key] * (1 - p_influence)
                        + research_relevances[program][endline_key] * p_influence
                    )
                    program_params["mainline"][
                        during_program_key
                    ] = research_relevances[program][endline_key]

                    program_params["mainline_cf"][
                        after_program_key
                    ] = research_relevances[program][baseline_key]
                    program_params["mainline_cf"][
                        during_program_key
                    ] = research_relevances[program][baseline_key]

            influence_params = {**influence_params, program: program_params}

        program_functions, program_data = so.get_program_data(
            programs=programs,
            default_parameters=influence_params,
            master_functions=master_functions,
            n_sim=n_sim,
            time_points=time_points,
        )

        for program in programs:
            for participant_type in [
                "participant",
                "attendee",
                "contender",
                "student_undergrad_via_phd",
                "student_undergrad_not_via_phd",
                "student_phd",
            ]:
                try:
                    qarys = program_data[program]["qarys_" + participant_type].mean()
                    qarys_cf = program_data[program][
                        "qarys_" + participant_type + "_cf"
                    ].mean()
                    budget = influence_params[program]["mainline"]["target_budget"]

                    results.append(
                        [
                            program,
                            participant_type,
                            p_influence,
                            qarys,
                            qarys_cf,
                            budget,
                        ]
                    )
                except:
                    pass

    results_df = pd.DataFrame(
        results,
        columns=[
            "program",
            "participant_type",
            "p_influence",
            "qarys",
            "qarys_cf",
            "target_budget",
        ],
    )

    results_df["qarys_diff"] = results_df["qarys"] - results_df["qarys_cf"]
    results_df["cost_effectiveness"] = results_df["qarys_diff"] / (
        results_df["target_budget"] / (1 * M)
    )

    return results_df


def plot_horizontal_budget_qarys_ce(
    df,
    program_colors=None,
    program_type="professional",
    title=None,
    xlabel=None,
    ylabel1=None,
    ylabel2=None,
    legend_pos="lower right",
    use_pseudo_log_scale_x=False,
    use_pseudo_log_scale_y=False,
    linthresh_x=0.0001,
    linthresh_y=0.0001,
    legend_column=0,
):
    """
    Plots QARYS Difference and Cost-Effectiveness against p_influence for attendees and contenders for different programs.

    Args:
        df (pandas.DataFrame): A DataFrame containing the data to be plotted.
        program_colors (dict): A dictionary mapping program names to colors.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel1 (str): Label for the y-axis on the left side (QARYS Difference).
        ylabel2 (str): Label for the y-axis on the right side (Cost-Effectiveness).
        legend_pos (str): Position of the legend.
        use_pseudo_log_scale (bool): Whether to use pseudo-log scale for y-axis or not.

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
    """

    programs = df["program"].unique()

    if program_colors is None:
        program_colors = plt.cm.get_cmap("tab10", len(programs))
        get_color = lambda program: program_colors[program]
    else:
        get_color = lambda program: program_colors[program]

    if program_type == "professional":
        participant_types = ["attendee", "contender"]
    elif program_type == "student":
        participant_types = [
            "student_undergrad_via_phd",
            "student_undergrad_not_via_phd",
            "student_phd",
        ]
    else:
        raise ValueError(f"Unknown program_type: {program_type}")

    fig, axes = plt.subplots(
        2,
        len(participant_types),
        figsize=(16, 16),
        sharey="row",
        gridspec_kw={
            "width_ratios": [1] * len(participant_types),
            "wspace": 0.1,
            "hspace": 0.1,
        },
    )

    for i, participant_type in enumerate(participant_types):
        for j, program in enumerate(programs):
            program_df = df[df["program"] == program]
            participant_df = program_df[
                program_df["participant_type"] == participant_type
            ]

            axes[0][i].plot(
                participant_df["p_influence"],
                participant_df["qarys_diff"],
                label=f"{help.prettify_label(program)} - {help.prettify_label(participant_type)}",
                color=get_color(program),
            )
            axes[1][i].plot(
                participant_df["p_influence"],
                participant_df["cost_effectiveness"],
                label=f"{help.prettify_label(program)} - {help.prettify_label(participant_type)}",
                color=get_color(program),
            )

        if use_pseudo_log_scale_x:
            axes[0][i].set_xscale("symlog", linthresh=linthresh_x)
            axes[1][i].set_xscale("symlog", linthresh=linthresh_x)

        if use_pseudo_log_scale_y:
            axes[0][i].set_yscale("symlog", linthresh=linthresh_y)
            axes[1][i].set_yscale("symlog", linthresh=linthresh_y)

        prettified_title = (
            help.prettify_label(participant_type, capitalize_each_word=False)
        )
        split_title = help.split_title(prettified_title, 20)
        axes[0][i].set_title(split_title)

        # Define a function that will format numbers as percentages.
        percent_formatter = FuncFormatter(
            lambda x, _: f"{x*100:.1g}%" if x < 0.01 else f"{x*100:.0f}%"
        )

        # Set the x-axis tick label format to percentage for the bottom two plots.
        axes[1][i].xaxis.set_major_formatter(percent_formatter)
        axes[0][i].get_xaxis().set_ticks([])

        axes[0][i].grid(alpha=0.3, which="both")
        axes[1][i].grid(alpha=0.3, which="both")

        for row in [0, 1]:
            axes[row][i].spines["top"].set_visible(False)
            axes[row][i].spines["right"].set_visible(False)
            axes[row][i].spines["left"].set_visible(False)
            axes[row][i].spines["bottom"].set_visible(False)

        axes[1][i].set_xlabel(
            help.split_title(xlabel, 30)
            if xlabel
            else help.split_title(
                "Probability shift research avenue relevance from 0 to 10", 30
            )
        )

    axes[0, 0].set_ylabel(
        ylabel1 if ylabel1 else "Benefit \n(counterfactual expected QARYs)"
    )
    axes[1, 0].set_ylabel(
        ylabel2 if ylabel2 else "Cost-effectiveness \n(expected QARYs per $1M)"
    )

    # Adjust the legend
    legend_patches = [
        mpatches.Patch(
            color=get_color(program.split(" - ")[0]),
            label=help.split_title(
                help.prettify_label(program, capitalize_each_word=True), 20
            ),
        )
        for program in programs
    ]
    axes[0, legend_column].legend(handles=legend_patches, loc=legend_pos)

    plt.tight_layout()

    return fig
