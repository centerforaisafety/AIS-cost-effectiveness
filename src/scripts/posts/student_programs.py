"""
Purpose: generate content for student field-building programs programs forum post
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Parameters
import parameters.atlas as p_a
import parameters.mlss as p_m
import parameters.student_club as p_sg
import parameters.undergraduate_stipends as p_us

# Models
import models.student_program as mfn_sp

# Sampling
import utilities.sampling.simulate_results as so

# Assumptions
import utilities.assumptions.research_avenues as ara

# Defaults
import utilities.defaults.plotting as d_plt

# Plotting
import utilities.plotting.helper_functions as help
import utilities.plotting.comparisons as comp
import utilities.plotting.impact_and_ability as ia
import utilities.plotting.influence as influence
import utilities.plotting.robustness as p_rob

# Common python packages
import numpy as np  # for sorting arrays
import pickle  # Save python objects

# Simulations
from squigglepy.numbers import K, M


"""
Pre-requisites
"""

# Parameters for simulating data
n_sim = 500 * K
time_points = np.concatenate(
    (
        np.arange(0.0, 0.1, 0.0002),
        np.arange(0.1, 1.5, 0.1),
        np.arange(1.5, 12.0, 0.01),
        np.arange(12.0, 61.0, 1.0),
    )
)
programs = ["atlas", "mlss", "student_club", "undergraduate_stipends"]
default_parameters = {
    "atlas": p_a,
    "mlss": p_m,
    "student_club": p_sg,
    "undergraduate_stipends": p_us,
}
master_functions = {
    "atlas": mfn_sp,
    "mlss": mfn_sp,
    "student_club": mfn_sp,
    "undergraduate_stipends": mfn_sp,
}

# Colors used throughout plots
program_colors = d_plt.program_colors_student
program_colors_multiple = d_plt.program_colors_multiple_student

# Parameters to display in tables
param_names_cost_effectiveness = ["target_budget", "qarys", "qarys_cf"]

param_names_cost = [
    "target_budget",
    "fixed_cost",
    "actual_variable_cost",
]

param_names_pipeline = [
    "p_pursue_ais",
    "p_pursue_ais_cf",
    "p_scientist_given_not_phd",
    "p_professor_given_not_phd",
    "p_engineer_given_not_phd",
    "p_scientist_equivalent_given_not_phd",
    "p_scientist_equivalent_given_not_phd_cf",
    "p_scientist_equivalent_via_not_phd",
    "p_scientist_equivalent_via_not_phd_cf",
    "p_phd_given_pursue_ais",
    "p_phd_given_pursue_ais_cf",
    "p_scientist_given_phd",
    "p_professor_given_phd",
    "p_engineer_given_phd",
    "p_scientist_equivalent_given_phd",
    "p_scientist_equivalent_given_phd_cf",
    "p_scientist_equivalent_via_phd",
    "p_scientist_equivalent_via_phd_cf",
    "p_scientist_equivalent",
    "p_scientist_equivalent_cf",
    "n_scientist_equivalent_student",
]

param_names = list(
    set(param_names_cost_effectiveness)
    | set(param_names_cost)
    | set(param_names_pipeline)
)


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
with open("output/data/student_programs/df_functions.pkl", "wb") as f:
    pickle.dump(df_functions, f)

with open("output/data/student_programs/df_params.pkl", "wb") as f:
    pickle.dump(df_params, f)

# Load data
with open("output/data/student_programs/df_functions.pkl", "rb") as f:
    df_functions = pickle.load(f)

with open("output/data/student_programs/df_params.pkl", "rb") as f:
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

# Load data on baseline programs
with open(
    "output/data/baseline_and_hypothetical_programs/baseline_programs.pkl", "rb"
) as f:
    baseline_programs = pickle.load(f)

help.formatted_markdown_table_cost_effectiveness(
    df_params_means,
    param_names_cost_effectiveness,
    help.format_number,
    bold_rows=[
        "Atlas",
        "MLSS",
        "Student Club",
        "Undergraduate Stipends",
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

# Number of participants plot
plot_student = comp.plot_overlapping_histograms(
    df_params_cost_and_participants,
    programs,
    program_colors,
    selected_variables=["n_student_undergrad", "n_student_phd"],
    log_scale=False,
    x_label_variable=True,
    single_legend_location="center right",
    remove_zeros=False,
    title="Number of pre-PhD and PhD students",
    subtitle="Across programs, given target budget allocated to programs",
)

plot_student.set_size_inches((10, 5))
plot_student.savefig(
    "output/plots/post_student-programs/build_n_student.png",
    dpi=300,
    bbox_inches="tight",
)

# Cost-effectiveness table
example_program = {
    "Simple Example Program": {
        "cost": 200 * K,
        "benefit": 11.6,
    }
}

help.formatted_markdown_table_cost_effectiveness_across_stages(
    [df_params_means_cost_and_participants],
    param_names,
    help.format_number,
    ["Simple Example Program", "Cost and Participants"],
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

# Pipeline plot
plot_pipeline = comp.plot_pipeline_probabilities(
    df_params_pipeline_and_equivalence,
    programs,
    program_colors,
    param_names_pipeline,
    selected_variables=[
        "p_pursue_ais",
        "p_pursue_ais_cf",
        "p_phd_given_pursue_ais",
        "p_phd_given_pursue_ais_cf",
        "p_scientist_equivalent_given_phd",
        "p_scientist_equivalent_given_phd_cf",
        "p_scientist_equivalent_given_not_phd",
        "p_scientist_equivalent_given_not_phd_cf",
        "p_scientist_equivalent",
        "p_scientist_equivalent_cf",
    ],
    log_scale=False,
    x_label_global="Probability",
    symlog_linthresh=1e-2,
    force_many_bins=True,
)

# Save the plot to a file
plot_pipeline.set_size_inches((12, 8.5))
plot_pipeline.savefig(
    "output/plots/post_student-programs/build_pipeline.png",
    dpi=300,
    bbox_inches="tight",
)

# N scientist equivalent plot
plot_n_scientist_equivalent = comp.plot_counterfactual_n_scientist_equivalent(
    df_params_pipeline_and_equivalence,
    programs,
    program_colors,
    selected_variables=[
        "n_scientist_equivalent_student_undergrad_during",
        "n_scientist_equivalent_student_undergrad_after",
        "n_scientist_equivalent_student_phd_during",
        "n_scientist_equivalent_student_phd_after",
    ],
    x_label_global="",
    pseudolog=False,
)

plot_n_scientist_equivalent.set_size_inches((12, 8.5))
plot_n_scientist_equivalent.savefig(
    "output/plots/post_student-programs/build_n_scientist_equivalent.png",
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
    ["Simple Example Program", "Cost and Participants", "Pipeline and Equivalence"],
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

# Ability plot
df = ia.extract_ability_parameters(default_parameters)
plot_ability_piecewise = ia.plot_mean_ability_piecewise(df, program_colors)

plot_ability_piecewise.set_size_inches((7, 6))
plot_ability_piecewise.savefig(
    "output/plots/post_student-programs/build_ability_piecewise.png",
    dpi=300,
    bbox_inches="tight",
)

# Relevance over time plot
variables_over_t = [
    "research_relevance",
]

plot_relevance_over_t = comp.plot_time_series_grid(
    df_functions_relevance_and_ability,
    variables_over_t,
    programs,
    participants=["student"],
    professions=["undergrad_via_phd", "undergrad_not_via_phd", "phd"],
    program_colors=program_colors,
    title="Research avenue relevance per scientist-equivalent participant over time",
    subtitle="",
    title_y_coordinate=1.13,
    x_min=-0.5,
    x_max=20,
    apply_custom_padding=True,
    legend_horizontal_anchor=1,
)

plot_relevance_over_t.set_size_inches((12, 3))
plot_relevance_over_t.savefig(
    "output/plots/post_student-programs/build_relevance_over_t.png",
    dpi=300,
    bbox_inches="tight",
)

# QARYs over time plot
variables_over_t = [
    "hours",
    "research_relevance",
    "qarys",
]

plot_qarys_over_t = comp.plot_time_series_grid(
    df_functions_relevance_and_ability,
    variables_over_t,
    programs,
    participants=["student"],
    professions=["undergrad_via_phd", "undergrad_not_via_phd", "phd"],
    program_colors=program_colors,
    title="QARYs per scientist-equivalent participant over time",
    subtitle="And selected component functions",
    title_y_coordinate=1.04,
    subtitle_y_coordinate=0.98,
    x_min=-0.5,
    x_max=20,
    apply_custom_padding=True,
    legend_horizontal_anchor=1,
)

plot_qarys_over_t.set_size_inches((12, 6))
plot_qarys_over_t.savefig(
    "output/plots/post_student-programs/build_relevance_qarys_over_t.png",
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
        "Simple Example Program",
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
    participants=["student"],
    professions=["undergrad_via_phd", "undergrad_not_via_phd", "phd"],
    program_colors=program_colors,
    title="Productivity, probability of staying in AI, and discounting over time",
    subtitle="",
    title_y_coordinate=1.0,
    x_min=-0.5,
    x_max=20,
    apply_custom_padding=True,
    legend_horizontal_anchor=1,
)

# Save the plot to a file
plot_remaining_functions_over_t.set_size_inches((12, 6))
plot_remaining_functions_over_t.savefig(
    "output/plots/post_student-programs/build_productivity_stayinai_timediscount.png",
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
    participants=["student"],
    professions=["undergrad_via_phd", "undergrad_not_via_phd", "phd"],
    program_colors=program_colors,
    title="QARYs per scientist-equivalent participant over time",
    subtitle="And component functions",
    title_y_coordinate=0.95,
    subtitle_y_coordinate=0.925,
    x_min=-0.5,
    x_max=20,
    apply_custom_padding=True,
    legend_horizontal_anchor=1,
)

# Save the plot to a file
plot_all_functions_over_t.set_size_inches((12, 15))
plot_all_functions_over_t.savefig(
    "output/plots/post_student-programs/build_all_time_functions.png",
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
    program_colors=program_colors,
    title="",
    alpha_robustness_lines=0,
)

# Save the plot to a file
plot_robustness_research_discount_rate.set_size_inches((6, 4))
plot_robustness_research_discount_rate.savefig(
    "output/plots/post_student-programs/robustness_research_discount_rate.png",
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
    "larger_student_costs",
    "smaller_student_costs",
    "better_job_prospects",
    "worse_job_prospects",
    "better_talent_spotting",
    "worse_talent_spotting",
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
    "output/plots/post_student-programs/robustness_scenarios.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Robustness: research relevance
"""

# Specify influence and relevance values for plot
p_influence_values = np.logspace(np.log10(0.0001), np.log10(1), 20)
research_relevances = {}

researcher_types = [
    "student_undergrad_via_phd",
    "student_undergrad_not_via_phd",
    "student_phd",
]
research_relevance_baseline = 0
research_relevance_endline = 100


def generate_research_relevances(program, researcher_types, baseline, endline):
    participant_types = {
        "atlas": researcher_types[0:2],
        "mlss": researcher_types[0:2],
        "student_club": researcher_types,
        "undergraduate_stipends": researcher_types[0:2],
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
    "student",
    xlabel="Probability shift research avenue relevance from 0 to 100",
    use_pseudo_log_scale_x=True,
    use_pseudo_log_scale_y=True,
    linthresh_y=1e-6,
    legend_column=2,
)

# Save the plot to a file
plot_influence.set_size_inches((10, 8))
plot_influence.savefig(
    "output/plots/post_student-programs/influence.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Value of research avenues
"""

# Plot
plot_research_avenue_relevance = ia.plot_impact_bar_chart(ara.impact_df)

# Save the plot to a file
plot_research_avenue_relevance.set_size_inches((15, 5))
plot_research_avenue_relevance.savefig(
    "output/plots/post_student-programs/research_avenue_relevance.png",
    dpi=300,
    bbox_inches="tight",
)
