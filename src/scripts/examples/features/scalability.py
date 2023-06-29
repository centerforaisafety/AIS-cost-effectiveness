"""
Example of how to assess scalability of programs to budget.
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

# Plotting
import utilities.plotting.budget as budget
from utilities.sampling.get_budget_sweep_data import get_budget_sweep_data

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
    (np.logspace(np.log10(0.0001), np.log10(10), 200), np.arange(10.0, 61.0, 1.0))
)
programs = ["tdc", "neurips_social", "neurips_workshop"]
default_parameters = {"tdc": p_p, "neurips_social": p_so, "neurips_workshop": p_w}
master_functions = {"tdc": mfn_pp, "neurips_social": mfn_pp, "neurips_workshop": mfn_pp}

# Colors used for different programs
program_colors = {"tdc": "green", "neurips_social": "purple", "neurips_workshop": "orange"}


"""
Simulate outcomes for different target budget values
"""

budget_values = np.concatenate(
    (
        [0],
        np.logspace(np.log10(2 * K), np.log10(50 * K), 15),
        np.arange(60 * K, 151 * K, 15 * K),
    )
)

df_budget = get_budget_sweep_data(
    programs,
    budget_values,
    default_parameters,
    master_functions,
    n_sim=n_sim,
    time_points=time_points,
    estimate_participants=True,
)


"""
Plot cost-effectiveness and number of participants for different target budget values
"""

# Outcomes as a function of budget
plot_budget_qarys = budget.plot_budget_qarys(
    df_budget,
    program_colors,
    title="Expected counterfactual QARYs and cost-effectiveness by budget",
    xlabel="Allocated budget amount",
    ylabel1="Expected counterfactual QARYs",
    ylabel2="Cost-effectiveness (QARYs per $1M)",
    legend_pos="lower right",
    use_pseudo_log_scale=True,
)

plot_budget_qarys.set_size_inches((12, 7))
plot_budget_qarys.savefig(
    "output/plots/examples/budget_qarys.png",
    dpi=300,
    bbox_inches="tight",
)

# Plot number of participants as a function of budget
plot_budget_n_participant = budget.plot_budget_n_participant(
    df_budget,
    program_colors,
    participant_types=["contender", "attendee"],
    title="Expected number of participants by budget",
    xlabel="Budget amount allocated to program (USD)",
    use_pseudo_log_scale=False,
    x_max_attendee=20 * K,
)

plot_budget_n_participant.set_size_inches((12, 7))
plot_budget_n_participant.savefig(
    "output/plots/examples/budget_n_participant.png",
    dpi=300,
    bbox_inches="tight",
)
