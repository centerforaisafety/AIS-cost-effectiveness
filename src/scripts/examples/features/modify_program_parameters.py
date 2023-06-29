"""
Example of how to modify parameters for a particular program
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Parameters
import parameters.mlss as p_m

# Simulations
import squigglepy as sq


"""
Modify parameters
"""

# 1. Modify program-specific parameters
students_per_instructor_modified = 10
students_per_instructor_modified_cf = students_per_instructor_modified
p_phd_given_pursue_ais_modified = sq.beta(20, 100)
p_phd_given_pursue_ais_modified_cf = sq.beta(10, 100)

# 2. List parameter instances to modify
defaults = ["mainline", "mainline_cf"]

# 3. Loop through parameter sets, modifying the parameters
for default in defaults:
    params = p_m.params[default]

    if default == "mainline":
        params["students_per_instructor"] = students_per_instructor_modified
        params["p_phd_given_pursue_ais"] = p_phd_given_pursue_ais_modified
    else:
        params["students_per_instructor"] = students_per_instructor_modified_cf
        params["p_phd_given_pursue_ais"] = p_phd_given_pursue_ais_modified_cf

    p_m.params[f'modification_name{"" if default == "mainline" else "_cf"}'] = params
