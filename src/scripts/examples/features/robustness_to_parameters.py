"""
Example of how to test and plot robustness of programs to particular parameters.
"""

"""
Imports
"""

import sys

sys.path.append("src")

# Parameters
import parameters.mlss as p_m
import parameters.student_group as p_sg
import parameters.undergraduate_stipends as p_ur

# Models
import models.student_program as mfn_sp

# Plotting
import utilities.plotting.robustness as rob

# Simulations
from squigglepy.numbers import K, M


"""
Inputs for data and plot
"""

# Set parameters for data
n_sim = 30 * K
time_points = range(0, 1 + 60)
programs = ["mlss", "student_group", "undergraduate_stipends"]
default_parameters = {"mlss": p_m, "student_group": p_sg, "undergraduate_stipends": p_ur}
master_functions = {
    "mlss": mfn_sp,
    "student_group": mfn_sp,
    "undergraduate_stipends": mfn_sp,
}

# Colors used throughout post for different programs
program_colors = {
    "mlss": "green",
    "student_group": "purple",
    "undergraduate_stipends": "orange",
}


"""
Simulate and plot results
"""

# 1. Specify the names and values of parameters to change, excluding default
robustness_changes = {"research_discount_rate": [-1, -0.5, -0.2, 0, 0.5]}

# 2. Simulate results for different values of the parameters
df_robustness_research_discount_rate = rob.process_robustness_data(
    programs,
    robustness_changes,
    default_parameters,
    master_functions,
    n_sim=100 * K,
)

# 3. Generate the plot
plot_robustness_research_discount_rate = rob.plot_robustness_multiple(
    df_robustness_research_discount_rate,
    program_colors,
    global_ylim=False,
    global_ylim_bottom=-1 * M,
    title="",
    axis_text_skip=1,
    ylim_buffer=0.5,
)

# 4. Save the plot to a file
plot_robustness_research_discount_rate.set_size_inches((6, 4))
plot_robustness_research_discount_rate.savefig(
    "output/plots/examples/robustness_research_discount_rate.png",
    dpi=300,
    bbox_inches="tight",
)
