"""
Purpose: generate content for introduction post
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

# Colors used throughout plots
program_colors_highlight_movers = d_plt.program_colors_all_highlight_movers
program_colors_categorical = d_plt.program_colors_all_categorical

# Parameters to display in tables
param_names_cost_effectiveness = ["target_budget", "qarys", "qarys_cf"]


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
with open("output/data/intro/df_functions.pkl", "wb") as f:
    pickle.dump(df_functions, f)

with open("output/data/intro/df_params.pkl", "wb") as f:
    pickle.dump(df_params, f)

# Load data
with open("output/data/intro/df_functions.pkl", "rb") as f:
    df_functions = pickle.load(f)

with open("output/data/intro/df_params.pkl", "rb") as f:
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
        "Atlas",
        "MLSS",
        "Student Club",
        "Undergraduate Stipends",
        "Trojan Detection Challenge",
        "NeurIPS ML Safety Social",
        "NeurIPS ML Safety Workshop",
    ],
    extra_programs=baseline_programs,
)


"""
Robustness: research discount rate
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
    n_sim=500 * K,
)

# Generate the plot
plot_robustness_research_discount_rate = p_rob.plot_robustness_multiple(
    df_robustness_research_discount_rate,
    program_colors=program_colors_highlight_movers,
    title="",
    alpha_robustness_default=0.45,
    alpha_robustness_points=0.25,
    alpha_robustness_lines=0,
    show_gridlines=True,
)

# Save the plot to a file
plot_robustness_research_discount_rate.set_size_inches((4.5, 4))
plot_robustness_research_discount_rate.savefig(
    "output/plots/post_intro/robustness_research_discount_rate.png",
    dpi=300,
    bbox_inches="tight",
)
