"""
Purpose: generate results for baseline and hypothetical programs
"""


"""
Imports
"""

import sys
import os

sys.path.append("src")

# Parameters
import parameters.tdc as p_p
import parameters.neurips_workshop as p_w

# Models
import models.professional_program as mfn_pp

# Sampling
import utilities.sampling.simulate_results as so

# Plotting
import utilities.plotting.helper_functions as help

# Common python packages
import numpy as np  # for sorting arrays
import pickle  # Save python objects

# Simulations
from squigglepy.numbers import K, M


"""
Pre-requisites
"""

# Set parameters for simulating results
n_sim = 1 * M
time_points = np.concatenate(
    (
        np.arange(0.0, 0.1, 0.0002),
        np.arange(0.1, 1.5, 0.1),
        np.arange(1.5, 12.0, 0.01),
        np.arange(12.0, 61.0, 1.0),
    )
)
programs = [
    "tdc",
    "neurips_workshop",
]
default_parameters = {
    "tdc": p_p,
    "neurips_workshop": p_w,
}
master_functions = {
    "tdc": mfn_pp,
    "neurips_workshop": mfn_pp,
}

# Parameters to display in tables
param_names_cost_effectiveness = ["target_budget", "qarys", "qarys_cf"]


"""
Cost-effectiveness of baseline programs
"""

# Cost-effectiveness baselines

os.makedirs('output/data/professional_programs', exist_ok = True)
with open("output/data/professional_programs/df_functions.pkl", "rb") as f:
    df_functions_professional = pickle.load(f)

with open("output/data/professional_programs/df_params.pkl", "rb") as f:
    df_params_professional = pickle.load(f)

scientist_trojans_cost = 500 * K
scientist_scientist_equivalence = 1
scientist_trojans_ability = 10
trojans_research_relevance = 10
scientist_trojans_benefit = (
    scientist_scientist_equivalence
    * scientist_trojans_ability
    * trojans_research_relevance
    * help.calculate_area(
        df_functions_professional["tdc"], end=1, researcher_type="scientist"
    )
)
scientist_trojans_cost_effectiveness = scientist_trojans_benefit / (
    scientist_trojans_cost / (1 * M)
)

phd_trojans_cost = (50 * K) * 5
phd_scientist_equivalence = 0.1
phd_trojans_ability = 10

phd_productivity_index_3 = (
    (df_functions_professional["tdc"]["time_point"] - 3).abs().idxmin()
)
phd_productivity_index_0 = 0
phd_productivity_3 = df_functions_professional["tdc"].loc[
    phd_productivity_index_3, "productivity_over_t_contender_phd"
]
phd_productivity_0 = df_functions_professional["tdc"].loc[
    phd_productivity_index_0, "productivity_over_t_contender_phd"
]

phd_trojans_benefit = (
    phd_scientist_equivalence
    * phd_trojans_ability
    * trojans_research_relevance
    * (
        help.calculate_area(
            df_functions_professional["tdc"],
            end=5,
            researcher_type="phd",
            productivity=phd_productivity_0,
        )
    )
)
phd_trojans_cost_effectiveness = phd_trojans_benefit / (
    phd_trojans_cost / (1 * M)
)

baseline_programs = {
    "Baseline:_Scientist_Trojans": [
        scientist_trojans_cost,
        scientist_trojans_benefit,
    ],
    "Baseline:_PhD_Trojans": [
        phd_trojans_cost,
        phd_trojans_benefit,
    ],
}

# Save data
with open(
    "output/data/baseline_and_hypothetical_programs/baseline_programs.pkl", "wb"
) as f:
    pickle.dump(baseline_programs, f)


"""
Cost-effectiveness of hypothetical hypothetical programs
"""

# New defaults for hypothetical programs
hypothetical_programs = ["power_aversion_prize", "cheaper_workshop"]
hypothetical_default_parameters = {"power_aversion_prize": p_p, "cheaper_workshop": p_w}
hypothetical_master_functions = {
    "power_aversion_prize": mfn_pp,
    "cheaper_workshop": mfn_pp,
}

# Alter parameters for hypothetical programs
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "target_budget"
] = (50 * K)
hypothetical_default_parameters["power_aversion_prize"].params["mainline_cf"][
    "target_budget"
] = (50 * K)

hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_during_program_contender_scientist"
] = 100
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_during_program_contender_professor"
] = 100
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_during_program_contender_engineer"
] = 100
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_during_program_contender_phd"
] = 100
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_contender_scientist"
] = (
    0.02
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
        "research_relevance_during_program_contender_scientist"
    ]
    + (1 - 0.02)
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline_cf"][
        "research_relevance_contender_scientist"
    ]
)
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_contender_professor"
] = (
    0.02
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
        "research_relevance_during_program_contender_professor"
    ]
    + (1 - 0.02)
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline_cf"][
        "research_relevance_contender_professor"
    ]
)
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_contender_engineer"
] = (
    0.02
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
        "research_relevance_during_program_contender_engineer"
    ]
    + (1 - 0.02)
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline_cf"][
        "research_relevance_contender_engineer"
    ]
)
hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
    "research_relevance_contender_phd"
] = (
    0.02
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline"][
        "research_relevance_during_program_contender_phd"
    ]
    + (1 - 0.02)
    * hypothetical_default_parameters["power_aversion_prize"].params["mainline_cf"][
        "research_relevance_contender_phd"
    ]
)

hypothetical_default_parameters["cheaper_workshop"].params["mainline"][
    "target_budget"
] = (35 * K)
hypothetical_default_parameters["cheaper_workshop"].params["mainline_cf"][
    "target_budget"
] = (35 * K)

# Call function that generates data
hypothetical_df_functions, hypothetical_df_params = so.get_program_data(
    programs=hypothetical_programs,
    default_parameters=hypothetical_default_parameters,
    master_functions=hypothetical_master_functions,
    n_sim=n_sim,
    time_points=time_points,
)

# Compute parameter means
hypothetical_df_params_means = help.compute_parameter_means(hypothetical_df_params)

# Rename programs
hypothetical_df_params_means = hypothetical_df_params_means.rename(columns={
    'power_aversion_prize': 'Hypothetical:_Power_Aversion_Prize',
    'cheaper_workshop': 'Hypothetical:_Cheaper_Workshop'
})

# Save data
with open(
    "output/data/baseline_and_hypothetical_programs/hypothetical_programs_means.pkl",
    "wb",
) as f:
    pickle.dump(hypothetical_df_params_means, f)
