"""
Example of how to generate summary statistics about programs.
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
import utilities.sampling.simulate_results as so  # for data

# Helper functions
import utilities.plotting.helper_functions as help  # for tables

# Common python packages
import pandas as pd  # for reading csv
import numpy as np  # for sorting arrays

# Simulations
from squigglepy.numbers import K, M


"""
Get data
"""

# Set parameters for data
n_sim = 30 * K
time_points = np.arange(0.0, 60.0, 1.0)
programs = ["tdc", "neurips_social", "neurips_workshop"]
default_parameters = {"tdc": p_p, "neurips_social": p_so, "neurips_workshop": p_w}
master_functions = {"tdc": mfn_pp, "neurips_social": mfn_pp, "neurips_workshop": mfn_pp}

# Call function that generates data
# This function returns two DataFrames: one with program functions and one with program parameters
df_functions, df_params = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
)


"""
Compute summary statistics
"""

# Compute parameter means for each program
means = {}
for program, df in df_params.items():
    means[program] = df.mean()

# Create a DataFrame with parameter means for each program
df_params_means = pd.DataFrame(means)
df_params_means.reset_index(inplace=True)
df_params_means.rename(columns={"index": "parameter"}, inplace=True)

# Specify the parameter names for the cost-effectiveness summary
param_names_cost_effectiveness = ["target_budget", "qarys", "qarys_cf"]

# Generate a formatted markdown table for cost-effectiveness summary
help.formatted_markdown_table_cost_effectiveness(
    df_params_means, param_names_cost_effectiveness, help.format_number
)
