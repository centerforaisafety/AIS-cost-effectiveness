"""
Purpose: example of how to explore CAIS's published results
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
import parameters.tdc as p_p
import parameters.neurips_social as p_so
import parameters.neurips_workshop as p_w

# Models
import models.student_program as mfn_sp
import models.professional_program as mfn_pp

# Sampling
import utilities.sampling.simulate_results as so

# Defaults
import utilities.defaults.plotting as d_plt

# Plotting
import utilities.plotting.helper_functions as help
import utilities.plotting.robustness as p_rob

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
n_sim = 100 * K
time_points = np.concatenate(
    (
        np.arange(0.0, 0.1, 0.0002),
        np.arange(0.1, 1.5, 0.1),
        np.arange(1.5, 12.0, 0.01),
        np.arange(12.0, 61.0, 1.0),
    )
)
programs = [
    "atlas",
    "mlss",
    "student_club",
    "undergraduate_stipends",
    "tdc",
    "neurips_social",
    "neurips_workshop",
]
default_parameters = {
    "atlas": p_a,
    "mlss": p_m,
    "student_club": p_sg,
    "undergraduate_stipends": p_us,
    "tdc": p_p,
    "neurips_social": p_so,
    "neurips_workshop": p_w,
}
master_functions = {
    "atlas": mfn_sp,
    "mlss": mfn_sp,
    "student_club": mfn_sp,
    "undergraduate_stipends": mfn_sp,
    "tdc": mfn_pp,
    "neurips_social": mfn_pp,
    "neurips_workshop": mfn_pp,
}


"""
1. Simulate results for default parameters
"""

# Call function that generates data
df_functions, df_params = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
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
2a. Modify global parameters
"""

# Narrow scientist-equivalence differences
scientist_equivalent_professor_modified = sq.to(2, 4)
scientist_equivalent_engineer_modified = 1 / scientist_equivalent_professor_modified
scientist_equivalent_phd_modified = 1

# Change age of research scientists using existing parameter value
params_tdc_mainline = default_parameters["tdc"].params["mainline"]
years_since_phd_scientist_modified = (
    params_tdc_mainline["years_since_phd_scientist"] + 5
)  # make scientists 5 years older

# Constant productivity
slope_productivity_modified = 0

# Dictionary of parameter modifications
param_modifications_global = {
    "scientist_equivalent_professor": scientist_equivalent_professor_modified,
    "scientist_equivalent_engineer": scientist_equivalent_engineer_modified,
    "scientist_equivalent_phd": scientist_equivalent_phd_modified,
    "years_since_phd_scientist": years_since_phd_scientist_modified,
    "slope_productivity": slope_productivity_modified,
}


"""
2b. Modify program-specific parameters: MLSS
"""

# Cheaper scholarships
variable_cost_per_student_modified = (10 * K + 7 * 2 * K) / 7
n_student_undergrad_scaling_parameter_slope_modified = (
    1 / variable_cost_per_student_modified
)

# Better job market odds conditional on not-PhD
params_mlss_mainline = default_parameters["mlss"].params["mainline"]
p_scientist_given_not_phd_modified_with_program = (
    params_mlss_mainline["p_scientist_given_not_phd"] * 3
)

# Dictionary of parameter modifications
param_modifications_mlss_cf = {
    "n_student_undergrad_scaling_parameter_slope": n_student_undergrad_scaling_parameter_slope_modified,
}

param_modifications_mlss = {
    **param_modifications_mlss_cf,
    "p_scientist_given_not_phd": p_scientist_given_not_phd_modified_with_program,
}


"""
2c. Modify program-specific parameters: NeurIPS Social
"""

# Smaller effect on research avenue relevance
params_neurips_social_mainline = default_parameters["neurips_social"].params["mainline"]
params_neurips_social_mainline_cf = default_parameters["neurips_social"].params[
    "mainline_cf"
]

relevance_scientist_mainline = params_neurips_social_mainline[
    "research_relevance_attendee_scientist"
]
relevance_scientist_mainline_cf = params_neurips_social_mainline_cf[
    "research_relevance_attendee_scientist"
]

relevance_modified = (
    relevance_scientist_mainline_cf
    + (relevance_scientist_mainline - relevance_scientist_mainline_cf) / 10
)

# Dictionary of parameter modifications
param_modifications_neurips_social_cf = {}

param_modifications_neurips_social = {
    **param_modifications_neurips_social_cf,
    "research_relevance_attendee_scientist": relevance_modified,
    "research_relevance_attendee_professor": relevance_modified,
    "research_relevance_attendee_engineer": relevance_modified,
    "research_relevance_attendee_phd": relevance_modified,
}


"""
3. Add modified parameter specifications back to params object
"""

for program in programs:
    # Get params objects
    params = default_parameters[program].params
    params_mainline = params["mainline"]
    params_mainline_cf = params["mainline_cf"]

    # Add global modifications
    params["modify_global_parameters"] = {
        **params_mainline,
        **param_modifications_global,
    }
    params["modify_global_parameters_cf"] = {
        **params_mainline_cf,
        **param_modifications_global,
    }

    # Add program-specific modifications, and/or global- _and_ program-specific modifications
    if program == "mlss":
        params["modify_program_parameters"] = {
            **params_mainline,
            **param_modifications_mlss,
        }
        params["modify_program_parameters_cf"] = {
            **params_mainline_cf,
            **param_modifications_mlss_cf,
        }
        params["modify_global_and_program_parameters"] = {
            **params["modify_global_parameters"],
            **param_modifications_mlss,
        }
        params["modify_global_and_program_parameters_cf"] = {
            **params["modify_global_parameters_cf"],
            **param_modifications_mlss_cf,
        }
    elif program == "neurips_social":
        params["modify_program_parameters"] = {
            **params_mainline,
            **param_modifications_neurips_social,
        }
        params["modify_program_parameters_cf"] = {
            **params_mainline_cf,
            **param_modifications_neurips_social_cf,
        }
        params["modify_global_and_program_parameters"] = {
            **params["modify_global_parameters"],
            **param_modifications_neurips_social,
        }
        params["modify_global_and_program_parameters_cf"] = {
            **params["modify_global_parameters_cf"],
            **param_modifications_neurips_social_cf,
        }
    else:
        params["modify_global_and_program_parameters"] = params[
            "modify_global_parameters"
        ]
        params["modify_global_and_program_parameters_cf"] = params[
            "modify_global_parameters_cf"
        ]


"""
4a. Simulate results for modified parameters: global parameters
"""

# Call function that generates data
df_functions_global, df_params_global = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification="modify_global_parameters",
)

# Compute parameter means
df_params_means_global = help.compute_parameter_means(df_params_global)

# Early cost-effectiveness table (for confirming that the above works)
help.formatted_markdown_table_cost_effectiveness(
    df_params_means_global, param_names_cost_effectiveness, help.format_number
)


"""
4b. Simulate results for modified parameters: program-specific parameters
"""

# Call function that generates data
df_functions_program, df_params_program = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification="modify_program_parameters",
)

# Compute parameter means
df_params_means_program = help.compute_parameter_means(df_params_program)

# Early cost-effectiveness table (for confirming that the above works)
help.formatted_markdown_table_cost_effectiveness(
    df_params_means_program, param_names_cost_effectiveness, help.format_number
)


"""
4c. Simulate results for modified parameters: global and program-specific parameters
"""

# Call function that generates data
df_functions_global_and_program, df_params_global_and_program = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    parameter_specification="modify_global_and_program_parameters",
)

# Compute parameter means
df_params_means_global_and_program = help.compute_parameter_means(
    df_params_global_and_program
)

# Early cost-effectiveness table (for confirming that the above works)
help.formatted_markdown_table_cost_effectiveness(
    df_params_means_global_and_program,
    param_names_cost_effectiveness,
    help.format_number,
)


"""
5. Cost-effectiveness table: across parameter specifications
"""

# Print table
help.formatted_markdown_table_cost_effectiveness_across_stages(
    [
        df_params_means,
        df_params_means_global,
        df_params_means_program,
        df_params_means_global_and_program,
    ],
    param_names_cost_effectiveness,
    help.format_number,
    [
        "Default parameters",
        "Global parameter modifications",
        "Program-specific parameter modifications",
        "Global and program-specific parameter modifications",
    ],
    bold_rows=[
        "Default parameters",
    ],
)


"""
6. Robustness to changes plot
"""

# Specify scenarios
scenarios = [
    "mainline",
    "modify_global_parameters",
    "modify_program_parameters",
    "modify_global_and_program_parameters",
]

# Get data
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

# Plot
plot_robustness_scenarios = p_rob.plot_benefit_cost_effectiveness_robustness_charts(
    data_robustness_processed, program_colors_categorical
)

# Save
plot_robustness_scenarios.set_size_inches((10, 5))
plot_robustness_scenarios.savefig(
    "output/plots/examples/robustness_scenarios.png",
    dpi=300,
    bbox_inches="tight",
)
