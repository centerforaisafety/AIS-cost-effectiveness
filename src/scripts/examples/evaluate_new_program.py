"""
Purpose: example of how to evaluate new program using existing models
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Models
import models.professional_program as mfn_pp

# Sampling
import utilities.sampling.simulate_results as so

# Assumptions
import utilities.assumptions.assumptions_baserates as adb

# Defaults
import utilities.defaults.plotting as d_plt

# Plotting
import utilities.plotting.helper_functions as help

# Common python packages
import numpy as np  # for sorting arrays

# Simulations
import squigglepy as sq
from squigglepy.numbers import K, M


"""
Pre-requisites
"""

# Colors used throughout post for different programs
program_colors_highlight_movers = d_plt.program_colors_all_highlight_movers
program_colors_categorical = d_plt.program_colors_all_categorical

# Parameters to display in tables
param_names_cost_effectiveness = ["target_budget", "qarys", "qarys_cf"]

# Parameters for simulating data
n_sim = 300 * K
time_points = np.concatenate(
    (
        np.arange(0.0, 0.1, 0.0002),
        np.arange(0.1, 1.5, 0.1),
        np.arange(1.5, 12.0, 0.01),
        np.arange(12.0, 61.0, 1.0),
    )
)
programs = [
    "phd_retreat",
]
master_functions = {
    "phd_retreat": mfn_pp,
}


"""
1. Specify parameter instances with and without the program
"""

cost_per_participant = 1 * K

params_mainline = {
    # Cost
    "target_budget": 20 * K,
    "fixed_hours_labor": sq.to(30, 300),
    "average_wage": 60,
    "split_variable_cost_event": 0.8,
    "split_variable_cost_award": 0,
    "sd_variable_cost_event": 100,
    "sd_variable_cost_award": 0,
    "sd_hours_labor": 10,
    "fixed_cost_other": 0,
    "event_max_capacity_per_1000": 20,
    # Number of people
    "n_attendee_scaling_parameter_gamma": 0,
    "n_attendee_scaling_parameter_slope": (1 / cost_per_participant),
    "n_attendee_scaling_parameter_intercept": 0,
    "fraction_attendee_scientist": 0,
    "fraction_attendee_professor": 0,
    "fraction_attendee_engineer": 0,
    # Pipeline and scientist-equivalence
    "p_scientist_given_phd": adb.p_scientist_given_phd,
    "p_professor_given_phd": adb.p_professor_given_phd * 2.1,
    "p_engineer_given_phd": adb.p_engineer_given_phd / 2,
    "scientist_equivalent_professor": adb.scientist_equivalent_professor,
    "scientist_equivalent_engineer": adb.scientist_equivalent_engineer,
    "scientist_equivalent_phd": adb.scientist_equivalent_phd,
    # Ability
    "ability_at_first_attendee_scientist": 1,
    "ability_at_pivot_attendee_scientist": 1,
    "ability_pivot_point_attendee_scientist": 10,
    "ability_at_first_attendee_professor": 1,
    "ability_at_pivot_attendee_professor": 1,
    "ability_pivot_point_attendee_professor": 10,
    "ability_at_first_attendee_engineer": 1,
    "ability_at_pivot_attendee_engineer": 1,
    "ability_pivot_point_attendee_engineer": 10,
    "ability_at_first_attendee_phd": 30,
    "ability_at_pivot_attendee_phd": 1,
    "ability_pivot_point_attendee_phd": 10,
    # Hours
    "hours_on_entry_per_attendee_scientist": 8 * 3,
    "hours_on_entry_per_attendee_professor": 8 * 3,
    "hours_on_entry_per_attendee_engineer": 8 * 3,
    "hours_on_entry_per_attendee_phd": 8 * 3,
    "hours_scientist_per_year": adb.hours_scientist_per_year,
    # Research avenue relevance
    "research_relevance_attendee_scientist": 0,
    "research_relevance_multiplier_after_phd_attendee_scientist": 0,
    "research_relevance_during_program_attendee_scientist": 0,
    "research_relevance_attendee_professor": 0,
    "research_relevance_multiplier_after_phd_attendee_professor": 0,
    "research_relevance_during_program_attendee_professor": 0,
    "research_relevance_attendee_engineer": 0,
    "research_relevance_multiplier_after_phd_attendee_engineer": 0,
    "research_relevance_during_program_attendee_engineer": 0,
    "research_relevance_attendee_phd": 13,
    "research_relevance_multiplier_after_phd_attendee_phd": 1,
    "research_relevance_during_program_attendee_phd": 100,
    # Productivity, staying in AI research, and time discounting
    "research_discount_rate": adb.research_discount_rate,
    "years_since_phd_scientist": 15 - adb.years_in_phd,
    "years_since_phd_professor": 15 - adb.years_in_phd,
    "years_since_phd_engineer": 10 - adb.years_in_phd,
    "years_since_phd_phd": 1 - adb.years_in_phd,
    "years_in_phd": adb.years_in_phd,
    "slope_productivity_life_cycle": adb.slope_productivity_life_cycle,
    "pivot_productivity_life_cycle": adb.pivot_productivity_life_cycle,
    "slope_staying_in_ai": adb.slope_staying_in_ai,
    "pivot_staying_in_ai": adb.pivot_staying_in_ai,
    "end_staying_in_ai": adb.end_staying_in_ai,
    # Flags
    "participant_contender": False,
    "participant_attendee": True,
}

additional_params_mainline_cf = {
    "p_professor_given_phd": adb.p_professor_given_phd * 2,
    "research_relevance_attendee_phd": 10,
    "research_relevance_during_program_attendee_phd": 10,
}

params_mainline_cf = {
    **params_mainline,
    **additional_params_mainline_cf,
}

params = {
    "mainline": params_mainline,
    "mainline_cf": params_mainline_cf,
}


"""
2. Obtain results
"""

# Call function that generates data
df_functions, df_params = so.get_program_data(
    programs=programs,
    default_parameters=params,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
)

# Compute parameter means
df_params_means = help.compute_parameter_means(df_params)

# Early cost-effectiveness table (for confirming that the above works)
help.formatted_markdown_table_cost_effectiveness(
    df_params_means, param_names_cost_effectiveness, help.format_number
)


"""
3. Summarize key parameter values
"""

# Define key parameters to summarize
key_parameters = [
    "target_budget",
    "fixed_cost",
    "actual_variable_cost",
    "n_attendee",
    "p_scientist_equivalent_given_phd",
    "p_scientist_equivalent_given_phd_cf",
    "n_scientist_equivalent_attendee_phd_during",
    "n_scientist_equivalent_attendee_phd_during_cf",
    "n_scientist_equivalent_attendee_phd_after",
    "n_scientist_equivalent_attendee_phd_after_cf",
    "qarys_per_attendee_phd",
    "qarys_per_attendee_phd_cf",
    "qarys",
    "qarys_cf",
]

# Create a subset DataFrame with rows matching key_parameters
subset_df = df_params_means[df_params_means["parameter"].isin(key_parameters)]

# Convert the subset DataFrame to a pretty Markdown table
markdown_table = subset_df.to_markdown(index=False)
print(markdown_table)
