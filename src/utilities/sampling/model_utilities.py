"""
Purpose: functions for sampling and merging parameters
"""


"""
Imports
"""

import squigglepy as sq
import numpy as np


"""
Sample given parameters
"""


def sample_params(params, n_sim=10000):
    """
    Sample from distributions and functions over time.

    Args:
        params (dict): Dictionary of parameters.
        n_sim (int): Number of simulations. Default is 1000.

    Output:
        tuple: Updated dictionaries for params and params_sampled.
    """

    # Initialize an empty dictionary to store sampled parameters
    params_sampled = {}

    # Sample parameters
    for key, value in params.items():
        if isinstance(
            value,
            (
                sq.distributions.NormalDistribution,
                sq.distributions.LognormalDistribution,
                sq.distributions.BetaDistribution,
                sq.distributions.ComplexDistribution,
                sq.distributions.GammaDistribution,
            ),
        ):
            params_sampled[key] = value @ n_sim
        else:
            params_sampled[key] = np.repeat(value, n_sim)

    # Return updated dictionaries
    return params, params_sampled


"""
Combine dictionary of sampled params with its cf params
"""


def merge_params_with_cf(params_sampled, params_sampled_cf):
    """
    Combine dictionary of sampled params with its cf params.

    Args:
        params_sampled (dict): Dictionary of sampled parameters.
        params_sampled_cf (dict): Dictionary of sampled cf parameters.

    Output:
        dict: Dictionary of merged parameters.
    """

    # Initialize an empty dictionary to store merged parameters
    params_sampled_merged = {}

    # Merge parameters
    for key in params_sampled.keys():
        params_sampled_merged[key] = params_sampled[key]
        cf_key = key + "_cf"
        params_sampled_merged[cf_key] = params_sampled_cf[key]

    # Return merged dictionary
    return params_sampled_merged
