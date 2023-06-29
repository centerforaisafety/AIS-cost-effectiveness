"""
Example of how to test robustness of cost-effectiveness estimates to various background parameters.
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
import parameters.undergraduate_stipends as p_ur

# Models
import models.student_program as mfn_sp

# Sampling
import utilities.sampling.simulate_results as so

# Plotting
import utilities.plotting.robustness as p_rob

# Common python packages
import numpy as np  # for sorting arrays

# Simulations
from squigglepy.numbers import K, M


"""
Inputs for data and plot
"""

# Set parameters for data
n_sim = 100 * K
time_points = np.concatenate(
    (
        np.arange(0.0, 1.5, 0.5),
        np.arange(1.5, 10.5, 0.01),
        np.arange(10.5, 15.0, 0.5),
        np.arange(15.0, 61.0, 1.0),
    )
)
programs = ["atlas", "mlss", "student_club", "undergraduate_stipends"]
default_parameters = {
    "atlas": p_a,
    "mlss": p_m,
    "student_club": p_sg,
    "undergraduate_stipends": p_ur,
}
master_functions = {
    "atlas": mfn_sp,
    "mlss": mfn_sp,
    "student_club": mfn_sp,
    "undergraduate_stipends": mfn_sp,
}

# Colors used for different programs
program_colors = {
    "atlas": "blue",
    "mlss": "green",
    "student_club": "purple",
    "undergraduate_stipends": "orange",
}


"""
Simulate and plot results
"""

# The parameters each scenario are specified at the bottom of each parameter script
# The parameter script calls perform_robustness_checks() from utilities.functions.robustness
# perform_robustness_checks() returns a dictionary of parameters for each scenario, specified as string
# get_scenario_data() then uses those parameters to generate data for each scenario

# 1. Specify scenarios
scenarios = [
    "mainline",
    "larger_difference_in_scientist_equivalence",
    "smaller_difference_in_scientist_equivalence",
    "larger_labor_costs",
    "smaller_labor_costs",
    "larger_fixed_costs",
    "smaller_fixed_costs",
    "better_job_prospects",
    "worse_job_prospects",
]

# 2. Simulate results
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

# 3. Process data for plotting
data_robustness_processed = p_rob.transform_data_robustness_to_df(data_robustness)

# 4. Plot
plot_robustness_scenarios = p_rob.plot_benefit_cost_effectiveness_robustness_charts(
    data_robustness_processed, program_colors
)

# 5. Save
plot_robustness_scenarios.set_size_inches((10, 5))
plot_robustness_scenarios.savefig(
    "output/plots/examples/robustness_scenarios.png",
    dpi=300,
    bbox_inches="tight",
)
