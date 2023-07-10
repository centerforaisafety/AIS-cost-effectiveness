"""
Purpose: generate content for field-building programs for professionals forum post
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Parameters
import parameters.tdc as p_p
import parameters.neurips_social as p_so
import parameters.neurips_workshop as p_w

# Models
import models.professional_program as mfn_pp

# Sampling
import utilities.sampling.simulate_results as so
from utilities.sampling.get_budget_sweep_data import get_budget_sweep_data

# Defaults
import utilities.defaults.plotting as d_plt

# Plotting
import utilities.plotting.helper_functions as help
import utilities.plotting.comparisons as comp
import utilities.plotting.influence as influence
import utilities.plotting.robustness as p_rob
import utilities.plotting.budget as budget

# Common python packages
import numpy as np  # for sorting arrays
import pickle  # Save python objects

# Simulations
from squigglepy.numbers import K, M


"""
Pre-requisites
"""

# Parameters for simulating data
n_sim = 1 * M
time_points = np.concatenate(
    (
        np.arange(0.0, 0.1, 0.0002),
        np.arange(0.1, 1.5, 0.1),
        np.arange(1.5, 12.0, 0.01),
        np.arange(12.0, 61.0, 1.0),
    )
)
programs = ["tdc", "neurips_social", "neurips_workshop"]
default_parameters = {"tdc": p_p, "neurips_social": p_so, "neurips_workshop": p_w}
master_functions = {"tdc": mfn_pp, "neurips_social": mfn_pp, "neurips_workshop": mfn_pp}

# Colors used throughout plots
program_colors = d_plt.program_colors_professional
program_colors_multiple = d_plt.program_colors_multiple_professional

# Parameters to display in tables
param_names_cost_effectiveness = ["target_budget", "qarys", "qarys_cf"]

param_names_cost = [
    "target_budget",
    "fixed_cost",
    "actual_variable_cost_event",
    "actual_variable_cost_award",
    "actual_variable_cost",
]

param_names = list(set(param_names_cost_effectiveness) | set(param_names_cost))


"""
Get results for default parameters
"""

# Call function that generates data
df_functions, df_params = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
)

# Save data
with open("output/data/professional_programs/df_functions.pkl", "wb") as f:
    pickle.dump(df_functions, f)

with open("output/data/professional_programs/df_params.pkl", "wb") as f:
    pickle.dump(df_params, f)

# Load data
with open("output/data/professional_programs/df_functions.pkl", "rb") as f:
    df_functions = pickle.load(f)

with open("output/data/professional_programs/df_params.pkl", "rb") as f:
    df_params = pickle.load(f)


"""
Compute parameter means
"""

# Compute parameter means
df_params_means = help.compute_parameter_means(df_params)

# Early cost-effectiveness table (for confirming that the above works)
help.formatted_markdown_table_cost_effectiveness(
    df_params_means, param_names_cost_effectiveness, help.format_number
)


"""
Cost-effectiveness table
"""

# Load data on hypothetical programs
with open(
    "output/data/baseline_and_hypothetical_programs/hypothetical_programs_means.pkl",
    "rb",
) as f:
    hypothetical_programs_means = pickle.load(f)

future_df_params_means = df_params_means.merge(
    hypothetical_programs_means, on="parameter"
)

future_df_params_means = future_df_params_means.rename(
    columns={
        "tdc": "Trojan_Detection_Challenge",
        "neurips_social": "NeurIPS_ML_Safety_Social",
        "neurips_workshop": "NeurIPS_ML_Safety_Workshop",
        "hypothetical:_power_aversion_prize": "Hypothetical:_Power_Aversion_Prize",
        "hypothetical:_cheaper_workshop": "Hypothetical:_Cheaper_Workshop",
    }
)

# Load data on baseline programs
with open(
    "output/data/baseline_and_hypothetical_programs/baseline_programs.pkl", "rb"
) as f:
    baseline_programs = pickle.load(f)

# Cost-effectiveness table
help.formatted_markdown_table_cost_effectiveness(
    future_df_params_means,
    param_names_cost_effectiveness,
    help.format_number,
    bold_rows=[
        "Trojan Detection Challenge",
        "NeurIPS ML Safety Social",
        "NeurIPS ML Safety Workshop",
    ],
    extra_programs=baseline_programs,
)


"""
Build-up: cost and participants
"""

# Preliminaries
parameter_specification = "build_cost_and_participants"

# Call function that generates data
(
    df_functions_cost_and_participants,
    df_params_cost_and_participants,
) = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification=parameter_specification,
)

# Cost table
df_params_means_cost_and_participants = help.compute_parameter_means(
    df_params_cost_and_participants
)

help.formatted_markdown_table(
    df_params_means_cost_and_participants,
    param_names_cost,
    param_names,
    help.format_number,
)

# Plot
plot_participant = comp.plot_overlapping_histograms(
    df_params_cost_and_participants,
    programs,
    program_colors,
    selected_variables=["n_contender", "n_attendee"],
    log_scale=False,
    x_label_global="",
    x_label_variable=True,
    single_legend_location="center",
    remove_zeros=False,
    title="Number of participants of each type",
    subtitle="Across programs, given default budget allocated to programs",
)

# Save the plot to a file
plot_participant.set_size_inches((10, 5))
plot_participant.savefig(
    "output/plots/post_professional-programs/build_participant.png",
    dpi=300,
    bbox_inches="tight",
)

# Cost-effectiveness table
example_program = {
    "Simple example program": {
        "cost": 100 * K,
        "benefit": 60,
    }
}

help.formatted_markdown_table_cost_effectiveness_across_stages(
    [df_params_means_cost_and_participants],
    param_names,
    help.format_number,
    ["Simple example program", "Cost and Participants"],
    bold_rows=["Cost and Participants"],
    extra_programs=example_program,
)


"""
Build-up: pipeline and equivalence
"""

# Preliminaries
parameter_specification = "build_pipeline_and_equivalence"


# Call function that generates data
(
    df_functions_pipeline_and_equivalence,
    df_params_pipeline_and_equivalence,
) = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification=parameter_specification,
)

# Plot, broken down by researcher type and participant
plot_participant_breakdown = comp.plot_n_scientist_equivalent(
    df_params_pipeline_and_equivalence,
    researcher_types=["scientist", "professor", "engineer", "phd_during", "phd_after"],
    linthresh=1e-1,
    title="Raw and scientist-equivalent number of participants",
)
plot_participant_breakdown.set_size_inches((15, 7))
plot_participant_breakdown.savefig(
    "output/plots/post_professional-programs/build_n_scientist_equivalent_breakdown.png",
    dpi=300,
    bbox_inches="tight",
)

# Cost-effectiveness table
df_params_means_pipeline_and_equivalence = help.compute_parameter_means(
    df_params_pipeline_and_equivalence
)

help.formatted_markdown_table_cost_effectiveness_across_stages(
    [df_params_means_cost_and_participants, df_params_means_pipeline_and_equivalence],
    param_names,
    help.format_number,
    ["Simple example program", "Cost and Participants", "Pipeline and Equivalence"],
    bold_rows=["Pipeline and Equivalence"],
    extra_programs=example_program,
)


"""
Build-up: relevance and ability
"""

# Preliminaries
parameter_specification = "build_relevance_and_ability"

# Call function that generates data
(
    df_functions_relevance_and_ability,
    df_params_relevance_and_ability,
) = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification=parameter_specification,
)

# Remaining time functions plot
variables_over_t = [
    "research_relevance",
]

plot_research_relevance_over_t = comp.plot_single_variable_time_series(
    df_functions_relevance_and_ability,
    "research_relevance",
    programs,
    participants=["contender", "attendee"],
    professions=["scientist", "professor", "engineer", "phd"],
    program_colors=program_colors_multiple,
    title="Research avenue relevance per scientist-equivalent participant over time",
    title_y_coordinate=1.13,
    subtitle="",
    ylim_padding=0.2,
    x_min=-0.5,
    x_max=20,
    legend_horizontal_anchor=1,
)

plot_research_relevance_over_t.set_size_inches((14, 3))
plot_research_relevance_over_t.savefig(
    "output/plots/post_professional-programs/build_relevance_over_t.png",
    dpi=300,
    bbox_inches="tight",
)

plot_zoomed_research_relevance_over_t = comp.plot_single_variable_time_series(
    df_functions_relevance_and_ability,
    "research_relevance",
    programs,
    participants=["contender", "attendee"],
    professions=["scientist", "professor", "engineer", "phd"],
    program_colors=program_colors_multiple,
    title="Research avenue relevance per scientist-equivalent participant over the 2 weeks following program start",
    subtitle="",
    title_y_coordinate=1.13,
    ylim_padding=0.2,
    x_min=0,
    x_max=0.04,
    legend_horizontal_anchor=1,
)

plot_zoomed_research_relevance_over_t.set_size_inches((14, 3))
plot_zoomed_research_relevance_over_t.savefig(
    "output/plots/post_professional-programs/build_zoomed_relevance_over_t.png",
    dpi=300,
    bbox_inches="tight",
)

# QARYs over time plot
variables_over_t = [
    "hours",
    "research_relevance",
    "qarys",
]

plot_all_functions_over_t = comp.plot_time_series_grid(
    df_functions_relevance_and_ability,
    variables_over_t,
    programs,
    title_y_coordinate=1.04,
    subtitle_y_coordinate=0.98,
    participants=["contender", "attendee"],
    professions=["scientist", "professor", "engineer", "phd"],
    program_colors=program_colors_multiple,
    title="QARYs per scientist-equivalent participant over time",
    subtitle="And selected component functions",
    x_min=-0.5,
    x_max=20,
    apply_custom_padding=True,
    legend_horizontal_anchor=1,
)

plot_all_functions_over_t.set_size_inches((14, 6))
plot_all_functions_over_t.savefig(
    "output/plots/post_professional-programs/build_relevance_qarys_over_t.png",
    dpi=300,
    bbox_inches="tight",
)

# Cost-effectiveness table
df_params_means_relevance_and_ability = help.compute_parameter_means(
    df_params_relevance_and_ability
)

help.formatted_markdown_table_cost_effectiveness_across_stages(
    [
        df_params_means_cost_and_participants,
        df_params_means_pipeline_and_equivalence,
        df_params_means_relevance_and_ability,
    ],
    param_names,
    help.format_number,
    [
        "Simple example program",
        "Cost and Participants",
        "Pipeline and Equivalence",
        "Ability and Relevance",
    ],
    bold_rows=["Ability and Relevance"],
    extra_programs=example_program,
)


"""
Build-up: remaining time functions
"""

# Preliminaries
parameter_specification = "build_remaining_time_functions"

# Call function that generates data
(
    df_functions_remaining_time_functions,
    df_params_remaining_time_functions,
) = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification=parameter_specification,
)

# Remaining time functions plot
variables_over_t = [
    "productivity",
    "p_staying_in_ai",
    "research_time_discount",
]

plot_remaining_functions_over_t = comp.plot_time_series_grid(
    df_functions_remaining_time_functions,
    variables_over_t,
    programs,
    participants=["contender", "attendee"],
    professions=["scientist", "professor", "engineer", "phd"],
    program_colors=program_colors_multiple,
    title="Productivity, probability of staying in AI, and discounting over time",
    subtitle="",
    x_min=-0.5,
    x_max=20,
    apply_custom_padding=True,
    legend_horizontal_anchor=1,
)

# Save the plot to a file
plot_remaining_functions_over_t.set_size_inches((14, 6))
plot_remaining_functions_over_t.savefig(
    "output/plots/post_professional-programs/build_productivity_stayinai_timediscount.png",
    dpi=300,
    bbox_inches="tight",
)

# All time functions plot
variables_over_t = [
    "hours",
    "research_relevance",
    "productivity",
    "p_staying_in_ai",
    "research_time_discount",
    "qarys",
]

plot_all_functions_over_t = comp.plot_time_series_grid(
    df_functions_remaining_time_functions,
    variables_over_t,
    programs,
    title_y_coordinate=0.95,
    subtitle_y_coordinate=0.925,
    participants=["contender", "attendee"],
    professions=["scientist", "professor", "engineer", "phd"],
    program_colors=program_colors_multiple,
    title="QARYs per scientist-equivalent participant over time",
    subtitle="And selected component functions",
    x_min=-0.5,
    x_max=20,
    symlog_ymin_research_relevance=0.3,
    symlog_ymin_qarys=0,
    symlog_yticks_research_relevance=[0.1, 0.3, 1, 3, 10, 30],
    symlog_yticks_qarys=[1, 3, 10, 30],
)

# Save the plot to a file
plot_all_functions_over_t.set_size_inches((14, 15))
plot_all_functions_over_t.savefig(
    "output/plots/post_professional-programs/build_all_time_functions.png",
    dpi=300,
    bbox_inches="tight",
)


# Cost-effectiveness table
df_params_means_remaining_time_functions = help.compute_parameter_means(
    df_params_remaining_time_functions
)

help.formatted_markdown_table_cost_effectiveness_across_stages(
    [
        df_params_means_cost_and_participants,
        df_params_means_pipeline_and_equivalence,
        df_params_means_relevance_and_ability,
        df_params_means_remaining_time_functions,
    ],
    param_names,
    help.format_number,
    [
        "Simple example program",
        "Cost and Participants",
        "Pipeline and Equivalence",
        "Ability and Relevance",
        "Productivity, Staying in AI, and Time Discounting",
    ],
    bold_rows=[
        "Productivity, Staying in AI, and Time Discounting",
    ],
    extra_programs=example_program,
)


"""
Budget sweep
"""

# Specify budget values for which we'll simulate outcomes
budget_values = np.concatenate(
    (
        np.arange(0, 20 * K, 2 * K),
        np.arange(20 * K, 40 * K, 5 * K),
        np.arange(40 * K, 60 * K, 10 * K),
        np.arange(60 * K, 161 * K, 20 * K),
    )
)

# Simulate results for different budget values
df_budget = get_budget_sweep_data(
    programs,
    budget_values,
    default_parameters,
    master_functions,
    n_sim=n_sim,
    time_points=time_points,
    estimate_participants=True,
)

# Save data
with open("output/data/professional_programs/df_budget.pkl", "wb") as f:
    pickle.dump(df_budget, f)

# Load data
with open("output/data/professional_programs/df_budget.pkl", "rb") as f:
    df_budget = pickle.load(f)


plot_budget_qarys = budget.plot_budget_qarys(
    df_budget,
    program_colors,
    title="Benefit and cost-effectiveness by budget",
    xlabel="Budget amount allocated to program (USD)",
    ylabel1="Benefit \n(counterfactual expected QARYs)",
    ylabel2="Cost-effectiveness \n(expected QARYs per $1M)",
    legend_pos="lower right",
    use_pseudo_log_y_scale=True,
)

# Save the plot to a file
plot_budget_qarys.set_size_inches((8, 6))
plot_budget_qarys.savefig(
    "output/plots/post_professional-programs/budget_qarys.png",
    dpi=300,
    bbox_inches="tight",
)

plot_budget_n_participant = budget.plot_budget_n_participant(
    df_budget,
    program_colors,
    participant_types=["contender", "attendee"],
    title="Expected number of participants by budget",
    xlabel="Budget amount allocated to program (USD)",
    use_pseudo_log_scale=False,
    x_max_attendee=None,
)
plot_budget_n_participant.set_size_inches((8, 6))
plot_budget_n_participant.savefig(
    "output/plots/post_professional-programs/budget_n_participant.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Robustness: parameters
"""

# Specify the names and values of parameters to change, excluding default
robustness_changes = {"research_discount_rate": np.delete(np.arange(-0.5, 0.6, 0.1), 7)}

robustness_research_discount_rate = {
    key: value
    for key, value in robustness_changes.items()
    if key == "research_discount_rate"
}

# Simulate results for different values of the parameters
df_robustness_research_discount_rate = p_rob.process_robustness_data(
    programs,
    robustness_research_discount_rate,
    default_parameters,
    master_functions,
    n_sim=n_sim,
)

# Generate the plot
plot_robustness_research_discount_rate = p_rob.plot_robustness_multiple(
    df_robustness_research_discount_rate,
    program_colors,
    global_ylim=False,
    global_ylim_bottom=-1 * M,
    title="",
    alpha_robustness_lines=0,
    axis_text_skip=1,
    ylim_buffer=0.5,
)

# Save the plot to a file
plot_robustness_research_discount_rate.set_size_inches((6, 4))
plot_robustness_research_discount_rate.savefig(
    "output/plots/post_professional-programs/robustness_research_discount_rate.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Robustness: scenarios
"""

# Specify scenarios
scenarios = [
    "mainline",
    "larger_difference_in_scientist_equivalence",
    "smaller_difference_in_scientist_equivalence",
    "larger_labor_costs",
    "smaller_labor_costs",
    "larger_fixed_costs",
    "smaller_fixed_costs",
    "more_researchers_per_entry",
    "fewer_researchers_per_entry",
    "more_senior_contender_composition",
    "less_senior_contender_composition",
    "more_senior_attendee_composition",
    "less_senior_attendee_composition",
    "more_hours_on_each_entry",
    "fewer_hours_on_each_entry",
]

# Simulate results
data_robustness = so.get_scenario_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    scenarios=scenarios,
    create_df_functions=False,
    share_params_cf=False,
)

# Process data for plotting
data_robustness_processed = p_rob.transform_data_robustness_to_df(data_robustness)

# Generate the plot
plot_robustness_scenarios = p_rob.plot_benefit_cost_effectiveness_robustness_charts(
    data_robustness_processed, program_colors
)

# Save the plot to a file
plot_robustness_scenarios.set_size_inches((10, 5))
plot_robustness_scenarios.savefig(
    "output/plots/post_professional-programs/robustness_scenarios.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Robustness: research relevance
"""

# Specify influence and relevance values for plot
p_influence_values = np.logspace(np.log10(0.0001), np.log10(1), 20)
research_relevances = {}

participant_types = ["contender", "attendee"]
researcher_types = (
    [type + "_professor" for type in participant_types]
    + [type + "_scientist" for type in participant_types]
    + [type + "_engineer" for type in participant_types]
    + [type + "_phd" for type in participant_types]
)

research_relevance_baseline = 0
research_relevance_endline = 10


def generate_research_relevances(program, researcher_types, baseline, endline):
    participant_types = {
        "tdc": [item for item in researcher_types if item.startswith("contender_")],
        "neurips_social": [
            item for item in researcher_types if item.startswith("attendee_")
        ],
        "neurips_workshop": researcher_types,
    }

    return {
        f"research_relevance_{participant_type}_{suffix}": value
        for participant_type in participant_types[program]
        for value, suffix in zip([baseline, endline], ["baseline", "endline"])
    }


research_relevances = {
    program: generate_research_relevances(
        program,
        researcher_types,
        research_relevance_baseline,
        research_relevance_endline,
    )
    for program in programs
}

# Simulate results
df = influence.simulate_benefit_under_differing_influence(
    default_params=default_parameters,
    p_influence_values=p_influence_values,
    programs=programs,
    research_relevances=research_relevances,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
)

# Generate the plot
plot_influence = influence.plot_horizontal_budget_qarys_ce(
    df,
    program_colors,
    "professional",
    xlabel="Probability shift research avenue relevance from 0 to 10",
    use_pseudo_log_scale_x=True,
    use_pseudo_log_scale_y=True,
    linthresh_y=1e-6,
    legend_column=1,
)

# Save the plot to a file
plot_influence.set_size_inches((10, 8))
plot_influence.savefig(
    "output/plots/post_professional-programs/influence.png",
    dpi=300,
    bbox_inches="tight",
)
