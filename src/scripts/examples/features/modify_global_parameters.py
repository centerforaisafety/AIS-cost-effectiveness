"""
Example of how to modify parameters across multiple programs
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

# Simulations
import squigglepy as sq


"""
Modify parameters
"""

# 1. Specify parameter modifications
scientist_equivalent_professor_modified = sq.to(1, 100)

# 2. List parameter sets to modify
parameter_sets = [p_p, p_so, p_w]

# 3. List parameter instances to modify for each program
defaults = ["mainline", "mainline_cf"]

# 4. Loop through parameter sets, modifying the parameter
for parameter_set in parameter_sets:
    for default in defaults:
        params = parameter_set.params[default]
        params[
            "scientist_equivalent_professor"
        ] = scientist_equivalent_professor_modified
        parameter_set.params[
            f'modification_name{"" if default == "mainline" else "_cf"}'
        ] = params
