"""
Example of how to plot comparisons between programs.
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
import utilities.plotting.comparisons as comp
import utilities.plotting.impact_and_ability as ia

# Common python packages
import numpy as np  # for sorting arrays

# Simulations
from squigglepy.numbers import K, M


"""
Inputs for data and plot
"""

# Set parameters for data
n_sim = 30 * K
time_points = np.concatenate(
    (
        np.arange(0.0, 1.5, 0.5),
        np.arange(1.5, 12.0, 0.01),
        np.arange(12.0, 61.0, 1.0),
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

# Colors used throughout post for different programs
program_colors = {
    "atlas": "#e7298a",
    "mlss": "#1b9e77",
    "student_club": "#984ea3",
    "undergraduate_stipends": "#d95f02",
}

"""
Simulate results
"""

# Call function that generates data
df_functions, df_params = so.get_program_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
)


"""
Plot number of participants
"""

# Plot
plot_n_student = comp.plot_overlapping_histograms(
    df_params,
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

# Save the plot to a file
plot_n_student.set_size_inches((10, 5))
plot_n_student.savefig(
    "output/plots/examples/n_student.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Plot ability
"""

df_ability = ia.extract_ability_parameters(default_parameters)
plot_ability_piecewise = ia.plot_mean_ability_piecewise(df_ability, program_colors)

plot_ability_piecewise.set_size_inches((7, 6))
plot_ability_piecewise.savefig(
    "output/plots/examples/ability_piecewise.png",
    dpi=300,
    bbox_inches="tight",
)


"""
Plot QARYs per scientist-equivalent participant over time
"""

# Variables to plot in rows
variables_over_t = [
    "hours",
    "research_relevance",
    "productivity",
    "p_staying_in_ai",
    "research_time_discount",
    "qarys",
]

# Plot
plot_all_functions_over_t = comp.plot_time_series_grid(
    df_functions,
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
plot_all_functions_over_t.set_size_inches((13, 15))
plot_all_functions_over_t.savefig(
    "output/plots/examples/QARYs_over_time.png",
    dpi=300,
    bbox_inches="tight",
)
